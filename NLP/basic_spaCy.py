import spacy


######## 방법1 ########
en_text = "A Dog Run back corner near spare bedrooms"
spacy_en = spacy.load('en')

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

print(tokenize(en_text))




######## 방법2 ########
## 다음처럼 spacy에서 내가 원하는 언어의 모델을 가져오고, 
nlp = spacy.load('en_core_web_sm')

## 다음처럼 문장을 nlp에 넘기기만 하면 끝 
doc = nlp('Apple is looking at buyin at U.K startup for $1 billion.')

print(type(doc))  ## 타입은, Doc고, 
print(doc)   ## 그냥 출력하면, 원래 문장이 그대로 나오고, 
print(list(doc))   ## 리스트로 변형하면, tokenize한 결과가 나오고 
print(type(doc[0]))   ## 리스트의 가장 앞에 있는 값은 Token이라는 타입 