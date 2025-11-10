def fieldDictBuild():
    fieldDict = dict.fromkeys([0, 1, 2, 3, 11, 12, 13, 14])

    # STEP 2 START
    fieldDict[11] = {
        "genhlth": 73,
        "bmi": (1533, 1536),
        "income": (124, 125),
        "education": 122,
    }
    fieldDict[12] = {
        "genhlth": 73,
        "bmi": (1644, 1647),
        "income": (116, 117),
        "education": 114,
    }
    fieldDict[13] = {
        "genhlth": 80,
        "bmi": (2192, 2195),
        "income": (152, 153),
        "education": 150,
    }
    fieldDict[14] = {
        "genhlth": 80,
        "bmi": (2247, 2250),
        "income": (152, 153),
        "education": 150,
    }
    # STEP 2 END

    return fieldDict


def getIncome(incomeString):
    if incomeString != "  ":
        income = int(incomeString)
    else:
        income = 9
    return income


def convertBMI(bmiString, shortYear):
    bmi = 0
    if shortYear == 0 and bmiString != "999":
        bmi = 0.1 * float(bmiString)
    if shortYear == 1 and bmiString != "999999":
        bmi = 0.0001 * float(bmiString)
    if 2 <= shortYear <= 10 and bmiString != "9999":
        bmi = 0.01 * float(bmiString)
    if shortYear > 10 and bmiString != "    ":
        bmi = 0.01 * float(bmiString)
    return bmi


def getEducation(educationString):
    if educationString != " ":
        education = int(educationString)
    else:
        education = 9
    return education


# STEP 8 START
def getHlth(hlthString):
    if hlthString != " ":
        genhlth = int(hlthString)
        if genhlth > 6:
            genhlth = -1
    else:
        genhlth = -1

    assert genhlth in (-1, 1, 2, 3, 4, 5, 6)
    return genhlth
