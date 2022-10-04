from fire import Fire


global FIRE
FIRE=-1

def main():
    global FIRE # but declaring global variable inside again, all the FIRE will be considered global one.
    FIRE=1
    printfire()
    print(FIRE) # refers to local one
    return


def printfire():
    global FIRE
    print(FIRE) # refers to global one
    FIRE=1 + FIRE # refers to local one unless L16 defines FIRE to be in a global scope
    return


if __name__ == '__main__':
    Fire(main)
