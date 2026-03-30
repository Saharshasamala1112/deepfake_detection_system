def postprocess(output):
    score = float(output)

    if score > 0.6:
        return "FAKE"
    elif score < 0.4:
        return "REAL"
    else:
        return "UNCERTAIN"