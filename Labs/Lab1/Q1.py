
# Question 1
num = int(input("Enter a number: "));

try:
    if num < 0:
        raise Exception("Input number should be positive")
    
    sum = 0
    for i in range(1,num+1):
        sum = sum + i

    print ("Sum of all numbers: ", sum)
    
except Exception as e:
    print("Exception caught: ", e)
