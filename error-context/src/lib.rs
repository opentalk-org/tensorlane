use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, LitStr, ReturnType, parse_macro_input, parse_quote};

#[proc_macro_attribute]
pub fn error_context(attr: TokenStream, item: TokenStream) -> TokenStream {
    let context = parse_macro_input!(attr as LitStr);
    let mut function = parse_macro_input!(item as ItemFn);

    let return_type = match &function.sig.output {
        ReturnType::Type(_, ty) => ty,
        ReturnType::Default => {
            return syn::Error::new_spanned(
                &function.sig,
                "error_context requires an anyhow::Result<T> return type",
            )
            .to_compile_error()
            .into();
        }
    };

    let body = &function.block;

    let wrapped_body = if function.sig.asyncness.is_some() {
        quote! {
            (async #body).await
        }
    } else {
        quote! {
            (|| -> #return_type #body)()
        }
    };

    function.block = parse_quote!({
        let result: #return_type = #wrapped_body;
        ::anyhow::Context::context(result, #context)
    });

    quote!(#function).into()
}
