from itertools import combinations

T = int(input())
for t in range(1, T+1):
    n, k = map(int, input().split())
    
    lst = list(range(1, 13))
    cnt = 0

    for comb in combinations(lst, n):
        if sum(comb) == k:
            cnt += 1


    print(f"#{t} {cnt}")

