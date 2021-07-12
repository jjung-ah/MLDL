# 생활데이터 표현하기
# 일주일간 유동인구 데이터(월요일~일요일)
import matplotlib.pyplot as plt 
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)   # 그래프 제목에 한글 표시하기

a = [242, 256, 237, 223, 81, 46]   # 리스트에 유동인구 데이터 초기화
print('A=', a)   # 출력하기

# 데이터의 합과 평균 구하기
n = len(a)   # 수열 a항 개수 구하기
my_sum = 0   # 합을 저장할 변수를 0으로 초기화
my_avg = 0   # 평균을 저장할 변수를 0으로 초기화
i = 0   # 수열 항의 인덱스, 파이썬은 첫 번째 수열 항의 인덱스는 0부터 시작함

for i in range(0, n):   # 인덱스 값은 0부터 시작하여 (n-1)까지 반복하기
    my_sum += a[i]   # 총합 구하기

my_avg = my_sum/n   # 평균 구하기
print('Total Sum : ', my_sum)   # 총합 출력하기
print('Total Average : ', my_avg)   # 평균 출력하기

x_data = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']   # x축에 표시할 제목 리스트에 저장
# 그래프 제목 붙이기
plt.title('일주일간 유동 인구수 데이터', fontsize=16)   # 큰 제목
plt.xlabel('요일', fontsize=12)   # x축 제목
plt.ylabel('유동 인구수', fontsize=12)   # y축 제목

# 꺾은선 그래프 그리기
plt.scatter(x_data, a)   # 꺽은선 그래프 그리기
plt.plot(x_data, a)
plt.show()


# 주중 데이터만으로 합과 평균 구하기
weekday_size = 5   # 주중이므로 5
weekday_sum = 0   # 합이 저장될 변수 초기화
weekday_avg = 0   # 평균이 저장될 변수 초기화

for i in range(0, weekday_size):
	weekday_sum += a[i]   # 주중 유동 인구수의 총합 구하기

weekday_avg = weekday_sum / weekday_size   # 주중 유동 인구수 평균 구하기

# 계산한 총합과 평균 출력하기
print('weekday Data=', a[0:5])   # 주중 데이터 출력하기
print('weekday Sum : ', weekday_sum)
print('weekday Average : ', weekday_avg)
# 그래프의 제목 붙이기
plt.title('주중 유동 인구수 데이터', fontsize=16)
plt.xlabel('요일', fontsize=12)
plt.ylabel('유동 인구수', fontsize=12)
# 꺾은선 그래프 그리기
plt.plot(x_data, a)
plt.scatter(x_data[0:weekday_size], a[0:weekday_size], c='red', edgecolor='none', s=50)
pli.show()

