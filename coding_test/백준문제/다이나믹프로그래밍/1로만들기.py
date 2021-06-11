# 정수 X에 사용할 수 있는 연산은 다음과 같이 세 가지 이다.

# X가 3으로 나누어 떨어지면, 3으로 나눈다.
# X가 2로 나누어 떨어지면, 2로 나눈다.
# 1을 뺀다.
# 정수 N이 주어졌을 때, 위와 같은 연산 세 개를 적절히 사용해서 1을 만들려고 한다. 연산을 사용하는 횟수의 최솟값을 출력하시오.

# 점화식 : dp(N) = min ( dp(N//3), dp(N//2), dp(N-1)) + 1

N = int(input())
dp_list = [0,0,1,1] # 0 ,1, 2, 3 의 최소 수 미리 저장

for i in range(4, N + 1) :
    # 먼저 1을 뺐을 경우 나오는 경우의 수 저장
    dp_list.append(dp_list[i-1] + 1)

    #2로 나누어질 경우 기존 1을 뺐을 경우의 수와 비교하여 최솟값 저장
    if i % 2 == 0 :
        dp_list[i] = min(dp_list[i], dp_list[i//2] + 1) # /쓰면 소수점나와서 에러, //로 몫 구하기

    #3으로 나누어질 경우 기존 경우의 수와 비교하여 최솟값 저장
    #여기서 2 또는 3으로 나누어질 경우 모든 경우를 봐야하므로 elif가 아닌 if로 설정
    if i % 3 == 0 :
        dp_list[i] = min(dp_list[i], dp_list[i//3] + 1)

print(dp_list[N])

# n = int(input())
# dp = [0 for _ in range(n+1)] #

# for i in range(2, n+1):
#     dp[i] = dp[i-1] + 1

#     if i%2 == 0 and dp[i] > dp[i//2] + 1 : 
#         dp[i] = dp[i//2]+1
        
#     if i%3 == 0 and dp[i] > dp[i//3] + 1 :
#         dp[i] = dp[i//3] + 1
        
# print(dp[n])



# # 실패 실패 실패 실패 실패 실패
# n = int(input())
# count = 0
# while True:
#     if n > 1:
#         if n%3 == 0:
#             n = n/3
#             count += 1
#         elif n%2 == 0:
#             n = n/2
#             count += 1
#         else:
#             n = n-1
#             count += 1
#     else:
#         break
# print(count)

# 실패 이유, 연산 횟수를 최소화 하려면 점화식을 이용해야 한다.