use std::{process::Child, time::Duration};

use crate::{
    config::{self, get_config, get_openai_endpoint, normalize_openai_endpoint},
    is_critical_err,
    program::Program,
    FuzzerError,
};
use async_openai::{
    config::{Config as LlmConfig, OpenAIConfig},
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse,
    },
    Client,
};
use reqwest::header::HeaderMap;
use secrecy::SecretString;
use eyre::Result;
use futures::future::join_all;
use once_cell::sync::OnceCell;
use serde::Serialize;
use serde_json::Value;

use super::Handler;

/// Token usage statistics structure
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub cached_tokens: u32,
}

impl TokenUsage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32, total_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens: 0,
        }
    }

    pub fn from_response(response: &CreateChatCompletionResponse) -> Self {
        if let Some(usage) = &response.usage {
            let cached_tokens = if let Some(details) = &usage.prompt_tokens_details {
                details.cached_tokens.unwrap_or(0)
            } else {
                0
            };
            Self {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cached_tokens,
            }
        } else {
            Self::default()
        }
    }

    pub fn from_raw_response(response: &Value) -> Self {
        let usage = response.get("usage").and_then(|v| v.as_object());

        let cached_tokens = usage
            .and_then(|u| u.get("prompt_tokens_details"))
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let get_u32 = |key: &str| {
            usage
                .and_then(|u| u.get(key))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32
        };

        Self {
            prompt_tokens: get_u32("prompt_tokens"),
            completion_tokens: get_u32("completion_tokens"),
            total_tokens: get_u32("total_tokens"),
            cached_tokens,
        }
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_tokens += other.cached_tokens;
    }
}

pub struct OpenAIHanler {
    _child: Option<Child>,
    rt: tokio::runtime::Runtime,
}

impl Default for OpenAIHanler {
    fn default() -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|_| panic!("Unable to build the openai runtime."));
        Self { _child: None, rt }
    }
}

impl Handler for OpenAIHanler {
    /// Generate `SAMPLE_N` programs by chatting with instructions.
    fn generate(&self, prompt: &super::prompt::Prompt) -> eyre::Result<Vec<Program>> {
        let start = std::time::Instant::now();
        let chat_msgs = prompt.to_chatgpt_message();
        let mut futures = Vec::new();
        for _ in 0..get_config().n_sample {
            let future = generate_program_by_chat(chat_msgs.clone());
            futures.push(future);
        }
        let results = self.rt.block_on(join_all(futures));

        let mut programs = Vec::new();
        let mut total_usage = TokenUsage::default();

        for result in results {
            if let Ok((program, usage)) = result {
                programs.push(program);
                total_usage.add(&usage);
            }
        }

        let elapsed = start.elapsed();
        log::info!("OpenAI Generate time: {}s", elapsed.as_secs());
        log::info!("Response Tokens\nCompletion Tokens: {}\nPrompt Tokens: {}\nTotal Tokens: {}\nCached Tokens: {}", 
                  total_usage.completion_tokens,
                  total_usage.prompt_tokens, 
                  total_usage.total_tokens,
                  total_usage.cached_tokens);

        Ok(programs)
    }
}

/// Chat-completions config.
///
/// `OpenAIConfig` in async-openai 0.28 sends `OpenAI-Beta: assistants=v2` on
/// every request. Gemini's OpenAI-compatible chat route is not the Assistants
/// API, and that header is not part of a chat completion call.
#[derive(Clone, Debug)]
struct ChatConfig {
    inner: OpenAIConfig,
}

impl LlmConfig for ChatConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = self.inner.headers();
        headers.remove("openai-beta");
        headers
    }

    fn url(&self, path: &str) -> String {
        self.inner.url(path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        self.inner.query()
    }

    fn api_base(&self) -> &str {
        self.inner.api_base()
    }

    fn api_key(&self) -> &SecretString {
        self.inner.api_key()
    }
}

/// Get the OpenAI interface client.
fn get_client() -> Result<&'static Client<ChatConfig>> {
    // read OpenAI API key form the env var (OPENAI_API_KEY).
    pub static CLIENT: OnceCell<Client<ChatConfig>> = OnceCell::new();
    let client = CLIENT.get_or_init(|| {
        // HTTP/1.1: reqwest's default HTTP/2 connection to Gemini's frontend
        // can complete with an empty body, which deserializes as EOF.
        let http_client = reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(300))
            .timeout(Duration::from_secs(600))
            .http1_only()
            .build()
            .unwrap();
        let endpoint = normalize_openai_endpoint(&get_openai_endpoint());
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let api_key = api_key.trim();
        if api_key.is_empty() {
            panic!("OPENAI_API_KEY not set");
        }
        log::info!("LLM chat URL: {endpoint}/chat/completions");
        let openai_config = OpenAIConfig::new()
            .with_api_base(endpoint)
            .with_api_key(api_key);
        let client = Client::with_config(ChatConfig {
            inner: openai_config,
        });

        client.with_http_client(http_client)
    });
    Ok(client)
}

/// Create a request for a chat prompt
fn create_chat_request(
    msgs: Vec<ChatCompletionRequestMessage>,
    stop: Option<String>,
) -> Result<CreateChatCompletionRequest> {
    let mut binding = CreateChatCompletionRequestArgs::default();
    let binding = binding.model(config::get_openai_model_name());

    let mut request = binding
        .messages(msgs)
        .temperature(config::get_config().temperature);
    if let Some(stop) = stop {
        request = request.stop(stop);
    }
    let request = request.build()?;
    Ok(request)
}

/// Get a response for a chat request.
///
/// The response is `serde_json::Value` so a Gemini `finish_reason` or usage
/// object that is not in OpenAI's enum still yields the message text.
async fn get_chat_response(request: CreateChatCompletionRequest) -> Result<Value> {
    let client: &Client<ChatConfig> = get_client().unwrap();
    for _retry in 0..config::RETRY_N {
        let response: std::result::Result<Value, OpenAIError> =
            client.chat().create_byot(&request).await;
        let response = response.map_err(eyre::Report::new);
        match is_critical_err(&response) {
            crate::Critical::Normal => {
                let response = response?;
                return Ok(response);
            }
            crate::Critical::NonCritical => {
                continue;
            }
            crate::Critical::Critical => return Err(response.err().unwrap()),
        }
    }
    Err(FuzzerError::RetryError(format!("{request:?}"), config::RETRY_N).into())
}

#[derive(Serialize)]
struct CreateChatCompletionRequestExtra {
    // https://serde.rs/attr-flatten.html
    #[serde(flatten)]
    pub request: CreateChatCompletionRequest, // original request type
    #[serde(flatten)]
    pub extra_body: serde_json::Value, // or this can be your custom type
}

impl CreateChatCompletionRequestExtra {
    pub fn with_deepseek_reason_body(request: CreateChatCompletionRequest) -> Self {
        let extra_body = serde_json::json!({
            "separate_reasoning": true,
            "chat_template_kwargs": {"thinking": true},
            "thinking": {"type": "enabled"}
        });
        //let extra_body = serde_json::json!({"reasoning": {"enabled": true}});
        Self {
            request,
            extra_body: extra_body,
        }
    }
}

/// Create a request for a chat prompt
fn create_reasoning_chat_request(
    msgs: Vec<ChatCompletionRequestMessage>,
    stop: Option<String>,
) -> Result<Value> {
    let mut binding: CreateChatCompletionRequestArgs = CreateChatCompletionRequestArgs::default();
    let binding = binding.model(config::get_openai_model_name());

    let mut request = binding
        .messages(msgs)
        .temperature(config::get_config().temperature);
    if let Some(stop) = stop {
        request = request.stop(stop);
    }
    let request = request.build()?;
    let request = CreateChatCompletionRequestExtra::with_deepseek_reason_body(request);
    let request = serde_json::to_value(request)?;
    Ok(request)
}

/// Get a response for a chat request
async fn get_reasoning_chat_response(request: Value) -> Result<Value> {
    let client: &Client<ChatConfig> = get_client().unwrap();

    for _retry in 0..config::RETRY_N {
        let response = client
            .chat()
            .create_byot(&request)
            .await
            .map_err(eyre::Report::new);
        match is_critical_err(&response) {
            crate::Critical::Normal => {
                let response = response?;
                return Ok(response);
            }
            crate::Critical::NonCritical => {
                continue;
            }
            crate::Critical::Critical => return Err(response.err().unwrap()),
        }
    }
    Err(FuzzerError::RetryError(format!("{request:?}"), config::RETRY_N).into())
}

pub async fn generate_program_by_chat(
    chat_msgs: Vec<ChatCompletionRequestMessage>,
) -> Result<(Program, TokenUsage)> {
    let (content, usage) = if get_config().deepseek_reasoning {
        let request = create_reasoning_chat_request(chat_msgs, None)?;
        let response = get_reasoning_chat_response(request).await?;
        let usage = TokenUsage::from_raw_response(&response);
        let content = chat_message_text(&response)?;
        (content, usage)
    } else {
        let request = create_chat_request(chat_msgs, None)?;
        let respond = get_chat_response(request).await?;
        let usage = TokenUsage::from_raw_response(&respond);
        let content = chat_message_text(&respond)?;
        (content, usage)
    };
    let content = strip_code_wrapper(&content);
    let program = Program::new(&content);
    Ok((program, usage))
}

fn chat_message_text(response: &Value) -> Result<String> {
    let message = response
        .pointer("/choices/0/message")
        .ok_or_else(|| eyre::eyre!("LLM response has no choices[0].message: {response}"))?;
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            return Ok(text.to_string());
        }
    }
    // Some OpenAI-compatible gateways return content as an array of parts.
    if let Some(parts) = message.get("content").and_then(Value::as_array) {
        let text: String = parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        if !text.is_empty() {
            return Ok(text);
        }
    }
    Err(eyre::eyre!("Empty response content: {message}"))
}

fn strip_code_prefix<'a>(input: &'a str, pat: &str) -> &'a str {
    let pat = String::from_iter(["```", pat]);
    if input.starts_with(&pat) {
        if let Some(p) = input.strip_prefix(&pat) {
            return p;
        }
    }
    input
}

/// strip the code wrapper that ChatGPT generated with code.
fn strip_code_wrapper(input: &str) -> String {
    let mut input = input.trim();
    let mut event = "";
    if let Some(idx) = input.find("```") {
        event = &input[..idx];
        input = &input[idx..];
    }
    let input = strip_code_prefix(input, "cpp");
    let input = strip_code_prefix(input, "CPP");
    let input = strip_code_prefix(input, "C++");
    let input = strip_code_prefix(input, "c++");
    let input = strip_code_prefix(input, "c");
    let input = strip_code_prefix(input, "C");
    let input = strip_code_prefix(input, "\n");
    if let Some(idx) = input.rfind("```") {
        let input = &input[..idx];
        let input = ["/*", event, "*/\n", input].concat();
        return input;
    }
    ["/*", event, "*/\n", input].concat()
}

#[cfg(test)]
mod tests {
    use async_openai::types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    };

    use super::*;

    #[test]
    fn test_normalize_gemini_endpoint() {
        assert_eq!(
            config::normalize_openai_endpoint(
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            config::normalize_openai_endpoint(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            ),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            config::normalize_openai_endpoint(
                "https://generativelanguage.googleapis.com/v1beta/openai/v1"
            ),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            config::normalize_openai_endpoint("https://api.openai.com/v1/"),
            "https://api.openai.com/v1"
        );
    }

    /// One chat completion against `OPENAI_ENDPOINT` / `OPENAI_MODEL_NAME` / `OPENAI_API_KEY`.
    ///
    /// ```text
    /// cargo test --lib request::openai::tests::test_llm_connectivity -- --ignored --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "live LLM call"]
    fn test_llm_connectivity() -> Result<()> {
        dotenv::dotenv().ok();
        config::init_openai_env();
        config::Config::init_test("libaom");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|_| panic!("Unable to build the openai runtime."));

        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("Reply with the single word pong.")
                .build()?
                .into(),
        ];
        let request = create_chat_request(messages, None)?;
        let client = get_client()?;
        let url = client.config().url("/chat/completions");
        println!("POST {url}");
        println!("model {}", config::get_openai_model_name());

        let outcome = rt.block_on(async {
            let result: std::result::Result<Value, OpenAIError> =
                client.chat().create_byot(&request).await;
            if let Err(err) = &result {
                println!("async-openai error: {err:#}");
                // The crate drops the response body after a parse failure.
                // Repeat the POST once so an empty Gemini body is visible.
                match reqwest::Client::builder()
                    .http1_only()
                    .build()
                    .unwrap()
                    .post(&url)
                    .bearer_auth(std::env::var("OPENAI_API_KEY").unwrap_or_default().trim())
                    .json(&request)
                    .send()
                    .await
                {
                    Ok(response) => {
                        let status = response.status();
                        let body = response.text().await.unwrap_or_default();
                        println!("raw HTTP {status}, {} bytes", body.len());
                        println!("{body}");
                    }
                    Err(err) => println!("raw POST failed: {err}"),
                }
            }
            result
        });

        let response = outcome.map_err(eyre::Report::new)?;
        println!("{response:#}");
        let content = chat_message_text(&response)?;
        assert!(
            !content.trim().is_empty(),
            "model returned empty content: {response}"
        );
        Ok(())
    }

    #[test]
    fn test_get_client() -> Result<()> {
        dotenv::dotenv().ok();
        config::init_openai_env();
        config::Config::init_test("libaom");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|_| panic!("Unable to build the openai runtime."));

        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Explain Rust's ownership system in simple terms.")
                .build()?
                .into(),
        ];

        let request = create_reasoning_chat_request(messages, None)?;
        let response = rt.block_on(get_reasoning_chat_response(request))?;
        println!("{:#?}", response);
        Ok(())
    }
}
