def is_palindrome(word):
    # 코드를 입력하세요.
    reversed_letter = ''
    for i in reversed(word):
        reversed_letter += i
    
    if word == reversed_letter:
        return True
    else:
        return False
    

        

# 테스트
print(is_palindrome("racecar"))
print(is_palindrome("stars"))
print(is_palindrome("토마토"))
print(is_palindrome("kayak"))
print(is_palindrome("hello"))