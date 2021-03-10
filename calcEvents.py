with open("./hieve_events.txt", "r+") as f:
    lines= f.readlines()
    data = []
    SuperSub = 0
    SubSuper = 0
    NoRel = 0
    Coref = 0

    for line in lines:

        line = line.replace("\n", "")
        line = line.split(",")
        # print(line)
        val1,val2, val3, val0 = 0,0,0,0
        for l in line:
            l = l.replace("{", "")
            l = l.replace("}", "")

            if "3: " in l:
                val3 = int(l[l.index("3: ")+3: ])
                NoRel+= int(l[l.index("3: ")+3: ])
            if "2: " in l:
                val2 = int(l[l.index("2: ")+3: ])
                Coref+= int(l[l.index("2: ")+3: ])
            if "1: " in l:
                val1 = int(l[l.index("1: ")+3: ])
                SuperSub += int(l[l.index("1: ")+3: ])
            if "0: " in l:
                val0 = int(l[l.index("0: ")+3: ])
                SubSuper += int(l[l.index("0: ")+3: ])
        ratio = (val0 + val1+ val2)
        print(ratio)
    # print(SuperSub)
    # print(SubSuper)
    # print(Coref)
    # print(NoRel)
