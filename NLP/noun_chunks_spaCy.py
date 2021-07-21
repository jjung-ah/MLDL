import spacy

doc = nlp("Autonomous cars shift insurance liability toward manufacturers")

## 특정 텍스트를 nlp에 넘기면 모두 해결되기는 하는데, 
## noun_chunks의 경우는 token 클래스도 아니고, Doc 클래스도 아니다. 
## Span이라는 클래스는 그냥 Doc와 비슷하다고 생각하면 된다, 일종의 복합어 개념.
noun_chunks = doc.noun_chunks
print(type(noun_chunks))
noun_chunk = list(noun_chunks)[0]
print(type(noun_chunk))
token = noun_chunk[0]
print(type(token))

print("=="*30)
print("""
Text: The original noun chunk text.
Root text: The original text of the word connecting the noun chunk to the rest of the parse.
Root dep: Dependency relation connecting the root to its head.
Root head text: The text of the root token's head.
""".strip())
print("=="*30)
str_format = "{:>25}"*4   # :<25 표현식을 사용하면 치환되는 문자열을 왼쪽으로 정렬하고 문자열의 총 자릿수를 25으로 맞출 수 있다. 반대로 오른쪽 정렬을 하고 싶으면 :>25를 사용하면 된다.
for chunk in doc.noun_chunks:
    print(str_format.format(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))
    