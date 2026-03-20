from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def translate(text, source_lang, target_lang):

    tokenizer.src_lang = source_lang

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=400
    )

    translated = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return translated[0]