use std::collections::{HashMap, HashSet};

use bytes::{BufMut, Bytes, BytesMut};

pub const PAD_SYMBOL: &str = "$";
pub const START_SYMBOL: &str = "<start/>";
pub const END_SYMBOL: &str = "<end/>";

pub const PUNCTUATION_SYMBOLS: &str = ";:,.!?¡¿—…\"«»“” ";
pub const LETTER_SYMBOLS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
pub const IPA_SYMBOLS: &str = concat!(
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœ",
    "ɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
);

pub fn default_styletts_symbols() -> Vec<String> {
    let mut symbols = vec![PAD_SYMBOL.to_string()];
    for text in [PUNCTUATION_SYMBOLS, LETTER_SYMBOLS, IPA_SYMBOLS] {
        symbols.extend(text.chars().map(|c| c.to_string()));
    }
    symbols.push(START_SYMBOL.to_string());
    symbols.push(END_SYMBOL.to_string());
    symbols
}

pub struct TextCleaner {
    symbol_index: HashMap<String, i64>,
    multichar_symbols: Vec<String>,
    chars_logged: HashSet<char>,
}

impl Default for TextCleaner {
    fn default() -> Self {
        Self::new(&default_styletts_symbols())
    }
}

impl TextCleaner {
    pub fn new(symbols: &[String]) -> Self {
        let mut symbol_index = HashMap::with_capacity(symbols.len());
        for (index, symbol) in symbols.iter().enumerate() {
            symbol_index.insert(symbol.clone(), index as i64);
        }

        let mut seen = HashSet::new();
        let mut multichar_symbols: Vec<String> = symbols
            .iter()
            .filter(|symbol| seen.insert(*symbol))
            .filter(|symbol| symbol.chars().count() > 1)
            .cloned()
            .collect();
        multichar_symbols.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()));

        Self {
            symbol_index,
            multichar_symbols,
            chars_logged: HashSet::new(),
        }
    }

    pub fn symbol_index(&self) -> &HashMap<String, i64> {
        &self.symbol_index
    }

    pub fn clean(&mut self, text: &str) -> Vec<i64> {
        let mut indexes = Vec::new();
        let mut position = 0;

        while position < text.len() {
            let rest = &text[position..];

            let symbol = self
                .multichar_symbols
                .iter()
                .find(|candidate| rest.starts_with(candidate.as_str()));

            if let Some(symbol) = symbol {
                indexes.push(self.symbol_index[symbol]);
                position += symbol.len();
                continue;
            }

            let character = rest.chars().next().expect("position is a char boundary");
            if let Some(index) = self.symbol_index.get(&character.to_string()) {
                indexes.push(*index);
            } else if self.chars_logged.insert(character) {
                tracing::warn!("character {character} not found in dictionary");
            }
            position += character.len_utf8();
        }

        indexes
    }
}

pub fn text_to_tensor(cleaner: &mut TextCleaner, boundary_token_id: i64, text: &str) -> Vec<i64> {
    let mut tokens = cleaner.clean(text);
    tokens.insert(0, boundary_token_id);
    tokens.push(boundary_token_id);
    tokens
}

pub fn text_to_tensor_bytes(
    cleaner: &mut TextCleaner,
    boundary_token_id: i64,
    text: &str,
) -> Bytes {
    let v = text_to_tensor(cleaner, boundary_token_id, text);
    let mut buff = BytesMut::with_capacity(8 * v.len());
    for t in v {
        buff.put_i64_le(t);
    }
    buff.freeze()
}

pub fn boundary_token_id(cleaner: &TextCleaner) -> anyhow::Result<i64> {
    cleaner
        .symbol_index()
        .get(PAD_SYMBOL)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("symbols are missing the pad symbol {PAD_SYMBOL:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cleaner() -> TextCleaner {
        TextCleaner::default()
    }

    #[test]
    fn matches_python_reference() {
        let mut tc = cleaner();
        assert_eq!(tc.clean("hello"), vec![50, 47, 54, 54, 57]);
        assert_eq!(tc.clean("ˈhɛloʊ"), vec![156, 50, 86, 54, 57, 135]);
        assert_eq!(tc.clean("<start/>hi<end/>"), vec![178, 50, 51, 179]);
    }

    #[test]
    fn wraps_in_boundary_tokens() {
        let mut tc = cleaner();
        let boundary = boundary_token_id(&tc).unwrap();
        assert_eq!(boundary, 0);
        assert_eq!(text_to_tensor(&mut tc, boundary, "hi"), vec![0, 50, 51, 0]);
        assert_eq!(text_to_tensor(&mut tc, boundary, ""), vec![0, 0]);
    }

    #[test]
    fn duplicate_symbol_keeps_last_index() {
        let symbols = default_styletts_symbols();
        assert_eq!(symbols.len(), 180);
        let mut tc = cleaner();
        assert_eq!(tc.symbol_index().len(), 179);
        assert_eq!(tc.clean("'"), vec![176]);
    }

    #[test]
    fn longest_match_wins() {
        let mut tc = cleaner();
        assert_eq!(tc.clean("<start/>"), vec![178]);
        assert_eq!(tc.clean("<start/"), vec![61, 62, 43, 60, 62]);
    }

    #[test]
    fn drops_unknown_characters() {
        let mut tc = cleaner();
        assert_eq!(tc.clean("a\u{1f600}b"), vec![43, 44]);
        assert_eq!(tc.clean("\u{1f600}"), Vec::<i64>::new());
    }

    #[test]
    fn does_not_normalize() {
        let mut tc = cleaner();
        assert_eq!(tc.clean("\u{e9}"), Vec::<i64>::new());
        assert_eq!(tc.clean("e\u{301}"), vec![47]);
        // the combining vertical line below IS a symbol
        assert_eq!(tc.clean("n\u{329}"), vec![56, 175]);
    }
}
