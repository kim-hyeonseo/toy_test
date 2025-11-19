T = int(input())

for t in range(1, T+1):
    p, at, bt  = map(int, input().split())

    al = bl = 1
    ar = br = p
    winner = '0'

    while al<=at or br>=bt:
        ac = (al + ar) // 2
        bc = (bl + br) // 2

        if ac == at and bc == bt:
            break

        elif ac == at:
            winner = 'A'
            break

        elif bc == bt:
            winner = 'B'
            break

        else : 

            if ac < at:
                al = ac

            else:
                ar = ac

            if bc < bt:
                bl = bc

            else:
                br = bc



    print(f"#{t} {winner}")