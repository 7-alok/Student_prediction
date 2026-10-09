import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------
# STUDENT PASS/FAIL PREDICTION SYSTEM
# ---------------------------------------

random.seed(42)

# Generate sample historical student data
X = []
y = []

for _ in range(500):
    study_hours = random.uniform(0, 10)
    attendance = random.uniform(30, 100)
    previous_marks = random.uniform(0, 100)
    assignments = random.randint(0, 10)

    # Simulate how these factors influence results
    score = (
        study_hours * 4
        + attendance * 0.20
        + previous_marks * 0.25
        + assignments * 1.5
    )

    passed = 1 if score >= 42 else 0

    X.append([
        study_hours,
        attendance,
        previous_marks,
        assignments
    ])
    y.append(passed)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# ---------------------------------------
# USER INTERFACE
# ---------------------------------------

print("\n" + "=" * 48)
print("       STUDENT SUCCESS PREDICTOR")
print("=" * 48)

print("\nEnter the student's details below:\n")

try:
    name = input("Student name: ").strip()
    if not name:
        name = "Student"

    hours = float(input("Daily study hours (0-24): "))
    attendance = float(input("Attendance percentage (0-100): "))
    marks = float(input("Previous exam marks (0-100): "))
    assignments = int(input("Assignments completed (0-10): "))

    # Validate inputs
    if not 0 <= hours <= 24:
        raise ValueError("Study hours must be between 0 and 24.")
    if not 0 <= attendance <= 100:
        raise ValueError("Attendance must be between 0 and 100.")
    if not 0 <= marks <= 100:
        raise ValueError("Marks must be between 0 and 100.")
    if not 0 <= assignments <= 10:
        raise ValueError("Assignments must be between 0 and 10.")

    student = [[hours, attendance, marks, assignments]]

    # Predict result and probability
    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]
    pass_probability = probabilities[1] * 100

    # Display report
    print("\n" + "=" * 48)
    print("            STUDENT REPORT")
    print("=" * 48)

    print(f"Student name       : {name}")
    print(f"Study hours/day    : {hours:.1f}")
    print(f"Attendance         : {attendance:.1f}%")
    print(f"Previous marks     : {marks:.1f}/100")
    print(f"Assignments        : {assignments}/10")

    print("\n" + "-" * 48)
    print(f"Model test accuracy: {accuracy * 100:.2f}%")
    print(f"Estimated pass probability: {pass_probability:.1f}%")

    if pass_probability >= 70:
        category = "HIGHER MODEL ESTIMATE"
        advice = "Keep up the good work and revise regularly!"
    elif pass_probability >= 40:
        category = "MODERATE MODEL ESTIMATE"
        advice = "Focus on revision and complete pending assignments."
    else:
        category = "LOWER MODEL ESTIMATE"
        advice = "Consider a study plan and seek support where needed."

    print(f"\nPerformance category: {category}")

    if prediction == 1:
        print("\nPREDICTION: PASS")
    else:
        print("\nPREDICTION: FAIL")

    print(f"Personalized advice: {advice}")
    print("=" * 48)

    print(
        "\nNote: This is an educational demo using "
        "synthetic data, not real student records."
    )
    print(
        "The probabilities and predictions are not "
        "validated measures of actual exam success."
    )

except ValueError as error:
    print(f"\nInvalid input: {error}")
    print("Please restart the program and enter valid values.")