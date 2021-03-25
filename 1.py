
def fun(**kwargs):
    print(kwargs)
    print(kwargs['d'])

c = {'a':1, 'b':7, 'dd':77}

fun(**c)