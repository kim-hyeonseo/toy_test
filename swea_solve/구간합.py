T = int(input())
for t in range(1, T+1):
    length, size = map(int, input().split())
    arr = list(map(int, input().split()))

    minV, maxV = 10**9, -10**9

    for i in range(length-size+1):
        tmp = sum(arr[i:i+size])


        minV = min(minV, tmp)
        maxV = max(maxV, tmp)

    print(f'#{t} {maxV - minV}')