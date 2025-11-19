T=int(input())
for t in range(1, T+1):
    n = int(input())    
    board = [[0]*11 for _ in range(11)]

    for _ in range(n):
        r1, c1, r2, c2, v = map(int, input().split())
        for i in range(r1, r2+1):
            for j in range(c1, c2+1):
                board[i][j] += v


    cnt = 0
    for i in range(len(board)):
        cnt+= board[i].count(3)

    print(f'#{t} {cnt}')

