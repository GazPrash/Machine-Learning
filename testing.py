j = 2
s = 0
while j <= 500:
    if j == 512 : break
    print(j)
    s += (500//j)
    j *= 2

print(s)