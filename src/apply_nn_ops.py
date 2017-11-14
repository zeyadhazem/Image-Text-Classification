# Takes in an an array of arrays
def apply_nn_operations(data):
    results = []
    for i in range(len(data)/3):
        if i % 1000 == 0:
            print(i)
        els = data[i*3:i*3+3]
        digits = []
        ops = []
        for e in els:
            symbol = e.index(max(e))
            if symbol < 10:
                digits.append(e)
            else:
                ops.append(e)
        # If there are 3 digits and no operators      
        if len(digits) >= 3:
            maxOp = 0
            newOp = 0
            for d in digits:
                score = max(d[9:])
                if score > maxOp:
                    newOp = d
                    maxOp = score
            digits.remove(newOp)
            ops.append(newOp)
        
        # If there's more than one operator do the opposite of above
        while len(ops) > 1:
            maxDig = 0
            newDig = 0
            for o in ops:
                score = max(o[:10])
                if score > maxDig:
                    newDig = o
                    maxDig = score  
            ops.remove(newDig)
            digits.append(newDig)
        
        # Max value out of 10,11
        op = ops[0].index(max(ops[0][9:]))
        # Max value out of 0-9
        dig1 = digits[0].index(max(digits[0][:10]))
        dig2 = digits[1].index(max(digits[1][:10]))
        if op == 10:
            results.append(dig1 + dig2)
        else:
            results.append(dig1 * dig2)               
    return results