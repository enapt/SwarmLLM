#[derive(Debug, Clone, PartialEq)]
pub(super) enum Token {
    /// Raw text between template tags
    Text(String),
    /// `{{ expr }}` — expression to evaluate and output
    Expr(String),
    /// `{% tag %}` — control flow statement
    Tag(String),
    /// `{%- tag %}` or `{% tag -%}` — tag with whitespace trimming
    TagTrimLeft(String),
    TagTrimRight(String),
    TagTrimBoth(String),
}

impl Token {
    pub(super) fn tag_content(&self) -> Option<&str> {
        match self {
            Token::Tag(s)
            | Token::TagTrimLeft(s)
            | Token::TagTrimRight(s)
            | Token::TagTrimBoth(s) => Some(s.trim()),
            _ => None,
        }
    }
}

// ── Tokenizer ──

pub(super) fn tokenize(template: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut rest = template;

    while !rest.is_empty() {
        // Find the next template marker
        if let Some(pos) = rest.find('{') {
            if pos > 0 {
                tokens.push(Token::Text(rest[..pos].to_string()));
                rest = &rest[pos..];
            }

            if rest.starts_with("{{") {
                // Expression: {{ ... }} with optional {{- / -}} trimming
                let end = rest.find("}}")?;
                let raw = &rest[2..end];

                // Check for {{- (trim left)
                let (trim_left, after_left) = if let Some(stripped) = raw.strip_prefix('-') {
                    (true, stripped)
                } else {
                    (false, raw)
                };
                // Check for -}} (trim right)
                let (trim_right, content) = if let Some(stripped) = after_left.strip_suffix('-') {
                    (true, stripped)
                } else {
                    (false, after_left)
                };

                if trim_left {
                    if let Some(Token::Text(ref mut t)) = tokens.last_mut() {
                        *t = t.trim_end().to_string();
                    }
                }

                tokens.push(Token::Expr(content.trim().to_string()));
                rest = &rest[end + 2..];

                if trim_right {
                    rest = rest.trim_start_matches([' ', '\t']);
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    } else if rest.starts_with("\r\n") {
                        rest = &rest[2..];
                    }
                }
            } else if rest.starts_with("{%") {
                // Tag: {% ... %}
                let end = find_tag_end(rest)?;
                let inner = &rest[2..end];

                // Check for whitespace trimming markers
                let trim_left = inner.starts_with('-');
                let trim_right = inner.ends_with('-');
                let content = inner
                    .trim_start_matches('-')
                    .trim_end_matches('-')
                    .to_string();

                let token = match (trim_left, trim_right) {
                    (true, true) => Token::TagTrimBoth(content),
                    (true, false) => Token::TagTrimLeft(content),
                    (false, true) => Token::TagTrimRight(content),
                    (false, false) => Token::Tag(content),
                };

                // Apply left trim (explicit `{%-`) or lstrip_blocks (default in HF Jinja2):
                // strip leading whitespace on the same line before a block tag.
                if trim_left {
                    if let Some(Token::Text(ref mut t)) = tokens.last_mut() {
                        *t = t.trim_end().to_string();
                    }
                } else {
                    // lstrip_blocks: strip spaces/tabs from start of line to the tag
                    if let Some(Token::Text(ref mut t)) = tokens.last_mut() {
                        if let Some(nl_pos) = t.rfind('\n') {
                            let after_nl = &t[nl_pos + 1..];
                            if after_nl.chars().all(|c| c == ' ' || c == '\t') {
                                t.truncate(nl_pos + 1);
                            }
                        } else if t.chars().all(|c| c == ' ' || c == '\t') {
                            // Entire text token is whitespace at start of template
                            t.clear();
                        }
                    }
                }

                tokens.push(token);
                rest = &rest[end + 2..];

                // Apply right trim (explicit `-%}`) OR trim_blocks (default in HF Jinja2):
                if trim_right {
                    rest = rest.trim_start_matches([' ', '\t']);
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    }
                } else {
                    // trim_blocks: strip exactly one newline after a block tag
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    } else if rest.starts_with("\r\n") {
                        rest = &rest[2..];
                    }
                }
            } else if rest.starts_with("{#") {
                // Comment: {# ... #}
                let end = rest.find("#}")?;
                rest = &rest[end + 2..];
            } else {
                // Lone '{' — treat as text
                tokens.push(Token::Text("{".to_string()));
                rest = &rest[1..];
            }
        } else {
            tokens.push(Token::Text(rest.to_string()));
            break;
        }
    }

    Some(tokens)
}

pub(super) fn find_tag_end(s: &str) -> Option<usize> {
    // s starts with "{%", find matching "%}"
    let inner = &s[2..];
    inner.find("%}").map(|pos| pos + 2)
}

// ── Evaluator ──
