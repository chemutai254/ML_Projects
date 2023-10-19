''' Lists - mutable
friends = ["Kevin", "Karen", "Jim", "Oscar", "Toby"]
friends[1] = "Mike"
friends.extend(["Nancy", "Laureen", "Waiyaki", "Makuyu", "Kameno"])
print(friends)
'''

''' Tuples - immutable
coordinates = [(4,5), (6,7),(8,9)]
print(coordinates[1])
'''

''' Functions
def sayhi(name):
    print(f"Hello user! {name}")
    
sayhi("Nancy")
'''

'''
def cube(num):
    return num * num * num
    
print(cube(3))
'''

# Dictionary
monthConversions = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}

print(monthConversions["Nov"])
print(monthConversions.get("Jul"))