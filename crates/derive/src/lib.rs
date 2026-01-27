//! Derive macros for JAX Privacy Rust.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, Ident, Index, Type};

/// Derive macro for `PyTree`.
///
/// This supports structs whose fields themselves implement
/// `jax_privacy_core::pytree::PyTree<Leaf = Tensor>`.
#[proc_macro_derive(PyTree)]
pub fn derive_pytree(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_pytree(&input) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

type SynResult<T> = Result<T, syn::Error>;

fn expand_pytree(input: &DeriveInput) -> SynResult<proc_macro2::TokenStream> {
    let name = &input.ident;
    let (field_accesses, field_inits, field_tys, field_count) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => expand_named_fields(fields.named.iter().collect())?,
            Fields::Unnamed(fields) => expand_unnamed_fields(fields.unnamed.iter().collect())?,
            Fields::Unit => (Vec::new(), Vec::new(), Vec::new(), 0usize),
        },
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "PyTree can only be derived for structs",
            ));
        }
    };

    let helper_mod = format_ident!("__jax_privacy_pytree_helper_{}", name);
    let leaf_ty = quote!(::jax_privacy_core::tensor::Tensor);
    let pytree_ty = quote!(::jax_privacy_core::pytree::PyTree);
    let treespec_ty = quote!(::jax_privacy_core::pytree::TreeSpec);
    let leaf_count_fn = quote!(::jax_privacy_core::pytree::leaf_count);

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_tys {
            where_clause
                .predicates
                .push(parse_quote!(#ty: #pytree_ty<Leaf = #leaf_ty>));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let spec_indices: Vec<Index> = (0..field_count).map(Index::from).collect();
    let struct_ctor = build_struct(name, &field_inits);

    let expanded = quote! {
        impl #impl_generics #pytree_ty for #name #ty_generics
        #where_clause
        {
            type Leaf = #leaf_ty;

            fn flatten(&self) -> (Vec<Self::Leaf>, #treespec_ty) {
                #[allow(non_snake_case)]
                mod #helper_mod {
                    use super::*;

                    pub fn build_spec(specs: &[#treespec_ty]) -> #treespec_ty {
                        match specs.len() {
                            0 => #treespec_ty::Option {
                                is_some: false,
                                child: Box::new(#treespec_ty::Leaf),
                            },
                            1 => specs[0].clone(),
                            _ => #treespec_ty::Tuple2(
                                Box::new(specs[0].clone()),
                                Box::new(build_spec(&specs[1..])),
                            ),
                        }
                    }
                }

                let mut leaves: Vec<Self::Leaf> = Vec::new();
                let mut specs: Vec<#treespec_ty> = Vec::with_capacity(#field_count);

                #(
                    let (mut field_leaves, field_spec) = #field_accesses.flatten();
                    leaves.append(&mut field_leaves);
                    specs.push(field_spec);
                )*

                let spec = #helper_mod::build_spec(&specs);
                (leaves, spec)
            }

            fn unflatten(spec: &#treespec_ty, leaves: Vec<Self::Leaf>) -> Self {
                #[allow(non_snake_case)]
                mod #helper_mod {
                    use super::*;

                    fn collect_specs(spec: &#treespec_ty, out: &mut Vec<#treespec_ty>) {
                        match spec {
                            #treespec_ty::Tuple2(a, b) => {
                                collect_specs(a, out);
                                collect_specs(b, out);
                            }
                            _ => out.push(spec.clone()),
                        }
                    }

                    pub fn specs_from(spec: &#treespec_ty, expected: usize) -> Vec<#treespec_ty> {
                        if expected == 0 {
                            return Vec::new();
                        }
                        let mut out = Vec::with_capacity(expected);
                        collect_specs(spec, &mut out);
                        assert_eq!(
                            out.len(),
                            expected,
                            "TreeSpec mismatch for {}: expected {} fields, got {}",
                            stringify!(#name),
                            expected,
                            out.len()
                        );
                        out
                    }
                }

                let specs = #helper_mod::specs_from(spec, #field_count);
                let leaves = leaves;
                let mut offset = 0usize;

                #(
                    let spec_i = &specs[#spec_indices];
                    let count_i = #leaf_count_fn(spec_i);
                    let end = offset + count_i;
                    assert!(
                        end <= leaves.len(),
                        "insufficient leaves for {} field {}",
                        stringify!(#name),
                        #spec_indices
                    );
                    let chunk = leaves[offset..end].to_vec();
                    let #field_inits = <#field_tys as #pytree_ty>::unflatten(spec_i, chunk);
                    offset = end;
                )*

                assert_eq!(
                    offset,
                    leaves.len(),
                    "unused leaves when unflattening {}",
                    stringify!(#name)
                );

                #struct_ctor
            }
        }
    };

    Ok(expanded)
}

fn expand_named_fields(
    fields: Vec<&syn::Field>,
) -> SynResult<(Vec<proc_macro2::TokenStream>, Vec<Ident>, Vec<Type>, usize)> {
    let mut accesses = Vec::with_capacity(fields.len());
    let mut inits = Vec::with_capacity(fields.len());
    let mut tys = Vec::with_capacity(fields.len());

    for field in fields {
        let ident = field
            .ident
            .clone()
            .ok_or_else(|| syn::Error::new_spanned(field, "expected named field"))?;
        accesses.push(quote!(self.#ident));
        inits.push(ident.clone());
        tys.push(field.ty.clone());
    }

    let count = inits.len();
    Ok((accesses, inits, tys, count))
}

fn expand_unnamed_fields(
    fields: Vec<&syn::Field>,
) -> SynResult<(Vec<proc_macro2::TokenStream>, Vec<Ident>, Vec<Type>, usize)> {
    let mut accesses = Vec::with_capacity(fields.len());
    let mut inits = Vec::with_capacity(fields.len());
    let mut tys = Vec::with_capacity(fields.len());

    for (i, field) in fields.into_iter().enumerate() {
        let idx = Index::from(i);
        let ident = format_ident!("__field_{}", i);
        accesses.push(quote!(self.#idx));
        inits.push(ident);
        tys.push(field.ty.clone());
    }

    let count = inits.len();
    Ok((accesses, inits, tys, count))
}

fn build_struct(name: &Ident, field_inits: &[Ident]) -> proc_macro2::TokenStream {
    if field_inits.is_empty() {
        return quote!(#name);
    }

    // Heuristic: if the first identifier matches our unnamed-field naming
    // convention, construct as a tuple struct.
    let is_tuple = field_inits[0].to_string().starts_with("__field_");

    if is_tuple {
        quote!(#name(#(#field_inits),*))
    } else {
        quote!(#name { #(#field_inits),* })
    }
}
