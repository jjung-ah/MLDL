def sol(L, x):
    c = 0
    L.sort()
    for i in L:
        if i <= x:
            c += 1
    print(c)
    L.insert(c, x)
    return L