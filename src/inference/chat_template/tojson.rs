//! `tojson` as HuggingFace `transformers` defines it.
//!
//! Every real tool-rendering chat template calls this filter, and the model's
//! author validated their template against `transformers`' version of it — so
//! that is the one to match, not Jinja2's builtin.
//!
//! Two differences from minijinja's builtin, both of which reached the model:
//!
//! - **It does not escape HTML.** minijinja rewrites `<`, `>`, `&` and `'`
//!   as `\u003c`, `\u003e`, `\u0026` and `\u0027`, because its filter is aimed at embedding
//!   JSON inside a web page. So a tool described as "the user's location"
//!   arrived as `the user\u0027s location`, and a schema mentioning `<` did the
//!   same — on every tool-carrying request to every model whose template
//!   renders tools through this filter. `transformers` overrides the builtin
//!   for exactly this reason and its code says so in a comment; llama.cpp's
//!   minja does not escape either.
//! - **It accepts `ensure_ascii`, `separators` and `sort_keys`.** minijinja's
//!   takes `indent` alone and calls `Kwargs::assert_all_used`, which turns any
//!   other keyword into `unknown keyword argument` — and an error in a filter
//!   fails the WHOLE render, not just that expression. GLM-4's template asks
//!   for `tojson(indent=4, ensure_ascii=False)`, so every GLM-4 request
//!   carrying tools was rendered by a fallback that knows nothing about the
//!   model's own `# 可用工具` framing. (minja rejects the same keyword with
//!   `Unknown argument ensure_ascii`, so llama.cpp has the bug too; that is a
//!   reason to check the reference, not to copy it.)
//!
//! The separators default is Python's, not serde_json's: `", "` and `": "` when
//! there is no indent, `","` and `": "` when there is. A bare `{{ x | tojson }}`
//! is what Qwen and Llama-3.1 use for tool schemas, and it now produces the same
//! bytes `transformers` produces.
//!
//! The single positional argument stays `indent`, as in Jinja2's builtin,
//! minijinja and minja. `transformers` reads that slot as `ensure_ascii`
//! instead, but no chat template passes it positionally — every one seen here
//! writes `tojson`, `tojson(indent=4)` or `tojson(indent=4, ensure_ascii=False)`
//! — so the compatible reading is the one that cannot silently reinterpret an
//! indent as a flag.

use minijinja::value::{Kwargs, Value};
use minijinja::{Error, ErrorKind};
use serde::Serialize;
use serde_json::ser::Formatter;
use std::io;

/// Python's `json.dumps` separators, which depend on whether an indent is set.
fn default_separators(indent: Option<usize>) -> (String, String) {
    if indent.is_some() {
        (",".to_string(), ": ".to_string())
    } else {
        (", ".to_string(), ": ".to_string())
    }
}

/// Read the `indent` argument, which may be positional or a keyword.
///
/// `true` means "pretty-print" and maps to two spaces, matching minijinja's
/// builtin, so a template that relied on that keeps working.
fn indent_width(indent: Option<Value>) -> Result<Option<usize>, Error> {
    let Some(val) = indent else { return Ok(None) };
    if val.is_none() || val.is_undefined() {
        return Ok(None);
    }
    if let Ok(flag) = bool::try_from(val.clone()) {
        return Ok(if flag { Some(2) } else { None });
    }
    Ok(Some(usize::try_from(val)?))
}

/// Read `separators=(item, key)`, a two-element sequence as in Python.
fn separators_from(val: Value) -> Result<(String, String), Error> {
    let parts: Vec<Value> = val
        .try_iter()
        .map_err(|_| {
            Error::new(
                ErrorKind::InvalidOperation,
                "tojson: separators must be a pair of strings",
            )
        })?
        .collect();
    if parts.len() != 2 {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson: separators must be a pair of strings",
        ));
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

pub fn tojson(value: Value, indent: Option<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let indent = match indent {
        Some(positional) => Some(positional),
        None => kwargs.get::<Option<Value>>("indent")?,
    };
    let indent = indent_width(indent)?;
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let (item_sep, key_sep) = match kwargs.get::<Option<Value>>("separators")? {
        Some(sep) if !sep.is_none() => separators_from(sep)?,
        _ => default_separators(indent),
    };
    kwargs.assert_all_used()?;

    let formatter = PythonFormatter {
        indent,
        level: 0,
        has_value: false,
        item_sep,
        key_sep,
        ensure_ascii,
    };

    // `sort_keys` is served by round-tripping through `serde_json::Value`,
    // whose map is a `BTreeMap` here and so is sorted by construction.
    //
    // It changes nothing at present, and that is worth stating plainly: every
    // map that reaches this filter is ALREADY key-sorted, because a tool
    // definition arrives as a `serde_json::Value` and `serde_json` is built
    // without `preserve_order`. The flag is honoured rather than rejected —
    // rejecting it is the defect this module exists to fix — and it will still
    // mean what it says if that ever changes.
    let rendered = if sort_keys {
        let sorted: serde_json::Value = serde_json::to_value(&value).map_err(json_error)?;
        write_json(&sorted, formatter)?
    } else {
        write_json(&value, formatter)?
    };

    Ok(Value::from_safe_string(rendered))
}

fn json_error(err: serde_json::Error) -> Error {
    Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
}

fn write_json<T: Serialize>(value: &T, formatter: PythonFormatter) -> Result<String, Error> {
    let mut out = Vec::<u8>::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
    value.serialize(&mut ser).map_err(json_error)?;
    String::from_utf8(out).map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
}

/// A `serde_json` formatter with Python's separators, indent and `ensure_ascii`.
///
/// Modelled on `serde_json::ser::PrettyFormatter` — the single `has_value` flag
/// is its trick, and it works because a nested container clears it on `begin_`
/// and the parent sets it again when that value ends.
struct PythonFormatter {
    indent: Option<usize>,
    level: usize,
    has_value: bool,
    item_sep: String,
    key_sep: String,
    ensure_ascii: bool,
}

impl PythonFormatter {
    fn newline_indent<W: ?Sized + io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let Some(width) = self.indent else {
            return Ok(());
        };
        writer.write_all(b"\n")?;
        for _ in 0..self.level * width {
            writer.write_all(b" ")?;
        }
        Ok(())
    }
}

impl Formatter for PythonFormatter {
    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.level += 1;
        self.has_value = false;
        writer.write_all(b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.level -= 1;
        if self.has_value {
            self.newline_indent(writer)?;
        }
        writer.write_all(b"]")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(self.item_sep.as_bytes())?;
        }
        self.newline_indent(writer)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.level += 1;
        self.has_value = false;
        writer.write_all(b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.level -= 1;
        if self.has_value {
            self.newline_indent(writer)?;
        }
        writer.write_all(b"}")
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(self.item_sep.as_bytes())?;
        }
        self.newline_indent(writer)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.key_sep.as_bytes())
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    /// The one hook `ensure_ascii` needs: string CONTENT passes through here,
    /// while quotes, backslashes and control characters are escaped by
    /// `write_char_escape` and are already ASCII.
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        let mut ascii_run = String::new();
        for ch in fragment.chars() {
            if ch.is_ascii() {
                ascii_run.push(ch);
                continue;
            }
            if !ascii_run.is_empty() {
                writer.write_all(ascii_run.as_bytes())?;
                ascii_run.clear();
            }
            // Python escapes astral characters as a surrogate PAIR, which is
            // what `encode_utf16` produces.
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                write!(writer, "\\u{unit:04x}")?;
            }
        }
        if !ascii_run.is_empty() {
            writer.write_all(ascii_run.as_bytes())?;
        }
        Ok(())
    }
}
