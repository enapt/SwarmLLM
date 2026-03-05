//! JSON grammar constraint for structured output.
//!
//! Tracks JSON parsing state during token-by-token generation and provides
//! a character-level validity check. Used to mask logits for tokens that would
//! produce invalid JSON.
//!
//! This is a lightweight pure-Rust approach that works with any model backend
//! (candle split models, llama.cpp, etc.) without requiring external grammar
//! libraries.

/// JSON grammar state machine for constraining token generation.
///
/// Tracks nesting depth, current context (object/array/string/etc), and
/// determines which characters are valid continuations.
#[derive(Debug, Clone)]
pub struct JsonGrammarState {
    /// Stack of nesting contexts.
    stack: Vec<JsonContext>,
    /// Current position within the current context.
    pos: Position,
    /// Whether we've completed a valid top-level JSON value.
    complete: bool,
    /// Total characters processed.
    char_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum JsonContext {
    Object,
    Array,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum Position {
    /// At the start, expecting a JSON value.
    Start,
    /// Inside a string literal.
    InString,
    /// After a backslash in a string (escape sequence).
    InStringEscape,
    /// Inside a number.
    InNumber,
    /// Inside a keyword (true/false/null).
    InKeyword { expected: &'static str, offset: usize },
    /// After a complete value, expecting comma/close/end.
    AfterValue,
    /// In an object, expecting a key (string).
    ExpectKey,
    /// After a key, expecting colon.
    ExpectColon,
    /// After colon, expecting a value.
    ExpectValue,
    /// After a comma in an object, expecting next key.
    ExpectNextKey,
    /// After a comma in an array, expecting next value.
    ExpectNextValue,
}

impl Default for JsonGrammarState {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammarState {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            pos: Position::Start,
            complete: false,
            char_count: 0,
        }
    }

    /// Check if the given string would be a valid continuation of the JSON.
    /// Returns true if ALL characters in the string are valid.
    pub fn is_valid_continuation(&self, text: &str) -> bool {
        let mut state = self.clone();
        for ch in text.chars() {
            if !state.accept_char(ch) {
                return false;
            }
        }
        true
    }

    /// Check if the current state represents a complete, valid JSON value.
    pub fn is_complete(&self) -> bool {
        self.complete && self.stack.is_empty()
    }

    /// Accept a single character, advancing the state machine.
    /// Returns true if the character is valid in the current state.
    pub fn accept_char(&mut self, ch: char) -> bool {
        self.char_count += 1;

        match self.pos {
            Position::Start | Position::ExpectValue | Position::ExpectNextValue => {
                self.accept_value_start(ch)
            }
            Position::InString => self.accept_in_string(ch),
            Position::InStringEscape => {
                // Accept any valid escape character
                if matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') {
                    self.pos = Position::InString;
                    true
                } else {
                    false
                }
            }
            Position::InNumber => self.accept_in_number(ch),
            Position::InKeyword { expected, offset } => {
                let expected_bytes = expected.as_bytes();
                if offset < expected_bytes.len() && ch == expected_bytes[offset] as char {
                    if offset + 1 == expected_bytes.len() {
                        self.pos = Position::AfterValue;
                        self.check_complete();
                    } else {
                        self.pos = Position::InKeyword {
                            expected,
                            offset: offset + 1,
                        };
                    }
                    true
                } else {
                    false
                }
            }
            Position::AfterValue => self.accept_after_value(ch),
            Position::ExpectKey | Position::ExpectNextKey => {
                match ch {
                    '"' => {
                        self.pos = Position::InString;
                        true
                    }
                    '}' if matches!(self.pos, Position::ExpectKey) => {
                        // Empty object or trailing comma (lenient)
                        self.stack.pop();
                        self.pos = Position::AfterValue;
                        self.check_complete();
                        true
                    }
                    c if c.is_whitespace() => true,
                    _ => false,
                }
            }
            Position::ExpectColon => match ch {
                ':' => {
                    self.pos = Position::ExpectValue;
                    true
                }
                c if c.is_whitespace() => true,
                _ => false,
            },
        }
    }

    fn accept_value_start(&mut self, ch: char) -> bool {
        match ch {
            '"' => {
                self.pos = Position::InString;
                true
            }
            '{' => {
                self.stack.push(JsonContext::Object);
                self.pos = Position::ExpectKey;
                true
            }
            '[' => {
                self.stack.push(JsonContext::Array);
                self.pos = Position::Start;
                true
            }
            't' => {
                self.pos = Position::InKeyword {
                    expected: "true",
                    offset: 1,
                };
                true
            }
            'f' => {
                self.pos = Position::InKeyword {
                    expected: "false",
                    offset: 1,
                };
                true
            }
            'n' => {
                self.pos = Position::InKeyword {
                    expected: "null",
                    offset: 1,
                };
                true
            }
            '-' | '0'..='9' => {
                self.pos = Position::InNumber;
                true
            }
            ']' if matches!(self.stack.last(), Some(JsonContext::Array)) => {
                self.stack.pop();
                self.pos = Position::AfterValue;
                self.check_complete();
                true
            }
            c if c.is_whitespace() => true,
            _ => false,
        }
    }

    fn accept_in_string(&mut self, ch: char) -> bool {
        match ch {
            '"' => {
                // String complete — context determines next state
                if matches!(self.stack.last(), Some(JsonContext::Object))
                    && matches!(
                        self.pos,
                        Position::InString
                    )
                {
                    // Could be a key or a value — check if we were expecting a key
                    // We need to track this better. For simplicity, after a string
                    // closes inside an object, check what comes next.
                    self.pos = Position::AfterValue;
                    self.check_complete();
                }
                self.pos = Position::AfterValue;
                self.check_complete();
                true
            }
            '\\' => {
                self.pos = Position::InStringEscape;
                true
            }
            // Control characters are not allowed in JSON strings
            c if c.is_control() => false,
            _ => true,
        }
    }

    fn accept_in_number(&mut self, ch: char) -> bool {
        match ch {
            '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => true,
            _ => {
                // Number ended — process this char as after-value
                self.pos = Position::AfterValue;
                self.check_complete();
                self.accept_after_value(ch)
            }
        }
    }

    fn accept_after_value(&mut self, ch: char) -> bool {
        match ch {
            ',' => {
                match self.stack.last() {
                    Some(JsonContext::Object) => self.pos = Position::ExpectNextKey,
                    Some(JsonContext::Array) => self.pos = Position::ExpectNextValue,
                    None => return false, // comma outside container
                }
                true
            }
            '}' => {
                if matches!(self.stack.last(), Some(JsonContext::Object)) {
                    self.stack.pop();
                    self.check_complete();
                    true
                } else {
                    false
                }
            }
            ']' => {
                if matches!(self.stack.last(), Some(JsonContext::Array)) {
                    self.stack.pop();
                    self.check_complete();
                    true
                } else {
                    false
                }
            }
            ':' if matches!(self.stack.last(), Some(JsonContext::Object)) => {
                // This handles the case where we just finished a key string
                self.pos = Position::ExpectValue;
                true
            }
            c if c.is_whitespace() => true,
            _ => false,
        }
    }

    fn check_complete(&mut self) {
        self.complete = self.stack.is_empty();
    }
}

/// Validate that a string is valid JSON. Returns Ok(()) or an error message.
pub fn validate_json(text: &str) -> Result<(), String> {
    serde_json::from_str::<serde_json::Value>(text)
        .map(|_| ())
        .map_err(|e| format!("Invalid JSON: {e}"))
}

/// Validate that a string is valid JSON conforming to a schema.
/// Uses basic type/required field validation (not full JSON Schema Draft 7).
pub fn validate_json_schema(text: &str, schema: &serde_json::Value) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("Invalid JSON: {e}"))?;
    validate_value_against_schema(&value, schema)
}

fn validate_value_against_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), String> {
    let schema_type = schema.get("type").and_then(|t| t.as_str());

    match schema_type {
        Some("object") => {
            let obj = value
                .as_object()
                .ok_or_else(|| "Expected object".to_string())?;

            // Check required fields
            if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                for req in required {
                    if let Some(key) = req.as_str() {
                        if !obj.contains_key(key) {
                            return Err(format!("Missing required field: {key}"));
                        }
                    }
                }
            }

            // Validate properties
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (key, prop_schema) in props {
                    if let Some(val) = obj.get(key) {
                        validate_value_against_schema(val, prop_schema)?;
                    }
                }
            }
            Ok(())
        }
        Some("array") => {
            let arr = value
                .as_array()
                .ok_or_else(|| "Expected array".to_string())?;
            if let Some(items_schema) = schema.get("items") {
                for (i, item) in arr.iter().enumerate() {
                    validate_value_against_schema(item, items_schema)
                        .map_err(|e| format!("Item {i}: {e}"))?;
                }
            }
            Ok(())
        }
        Some("string") => {
            if value.is_string() {
                Ok(())
            } else {
                Err(format!("Expected string, got {}", value_type_name(value)))
            }
        }
        Some("number" | "integer") => {
            if value.is_number() {
                Ok(())
            } else {
                Err(format!("Expected number, got {}", value_type_name(value)))
            }
        }
        Some("boolean") => {
            if value.is_boolean() {
                Ok(())
            } else {
                Err(format!("Expected boolean, got {}", value_type_name(value)))
            }
        }
        Some("null") => {
            if value.is_null() {
                Ok(())
            } else {
                Err(format!("Expected null, got {}", value_type_name(value)))
            }
        }
        _ => Ok(()), // Unknown or missing type — accept anything
    }
}

fn value_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_accepts_simple_object() {
        let mut state = JsonGrammarState::new();
        for ch in r#"{"key": "value"}"#.chars() {
            assert!(state.accept_char(ch), "failed at char '{ch}'");
        }
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_accepts_nested_object() {
        let mut state = JsonGrammarState::new();
        let json = r#"{"a": {"b": [1, 2, 3]}, "c": true}"#;
        for ch in json.chars() {
            assert!(state.accept_char(ch), "failed at char '{ch}'");
        }
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_accepts_array() {
        let mut state = JsonGrammarState::new();
        for ch in r#"[1, "two", null, false]"#.chars() {
            assert!(state.accept_char(ch), "failed at char '{ch}'");
        }
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_rejects_invalid_continuation() {
        let mut state = JsonGrammarState::new();
        // After a complete value at top level, no more chars
        for ch in r#"{"a": 1}"#.chars() {
            assert!(state.accept_char(ch));
        }
        assert!(state.is_complete());
        assert!(!state.accept_char('x'));
    }

    #[test]
    fn grammar_rejects_bare_text() {
        let mut state = JsonGrammarState::new();
        assert!(!state.accept_char('h')); // 'h' is not a valid JSON start
    }

    #[test]
    fn grammar_continuation_check() {
        let state = JsonGrammarState::new();
        assert!(state.is_valid_continuation("{"));
        assert!(state.is_valid_continuation(r#"{"key"#));
        assert!(!state.is_valid_continuation("hello"));
    }

    #[test]
    fn validate_json_valid() {
        assert!(validate_json(r#"{"a": 1}"#).is_ok());
        assert!(validate_json(r#"[1, 2, 3]"#).is_ok());
    }

    #[test]
    fn validate_json_invalid() {
        assert!(validate_json("not json").is_err());
        assert!(validate_json("{incomplete").is_err());
    }

    #[test]
    fn validate_schema_basic() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        });
        assert!(validate_json_schema(r#"{"name": "Alice", "age": 30}"#, &schema).is_ok());
        assert!(validate_json_schema(r#"{"age": 30}"#, &schema).is_err()); // missing required
        assert!(validate_json_schema(r#"{"name": 42}"#, &schema).is_err()); // wrong type
    }

    #[test]
    fn validate_schema_nested() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        });
        assert!(validate_json_schema(r#"{"items": ["a", "b"]}"#, &schema).is_ok());
        assert!(validate_json_schema(r#"{"items": [1, 2]}"#, &schema).is_err());
    }
}
