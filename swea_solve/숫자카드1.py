T  = int(input())
for t in range(1, T+1):
    length = int(input())
    input_str = input()

    cnt_list = [0] *10

    for num in input_str:
        cnt_list[int(num)] +=1
        
        
        
    max_cnt = max(cnt_list[::-1])
    max_value = 9 - cnt_list[::-1].index(max_cnt)

    print(f'#{t} {max_value} {max_cnt}' )
            	