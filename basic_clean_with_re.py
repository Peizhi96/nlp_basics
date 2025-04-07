import re
def clean_text(text):
  if not isinstance(text, str):
    return ""
  text = text.lower()
  #delete all non-alphanumeric characters 
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
  #process url
  text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
  #process html
  text = re.sub(r'<.*?>', ' ', text)
  #merge consecutive spaces
  text = re.sub(r'\s+', ' ', text).strip()
  return text