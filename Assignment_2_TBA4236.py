import numpy as np

def load_coordinates(file_path):
    coordinates = []
    with open(file_path, "r") as file:
        for line in file.readlines():
            x, y = line.split(",")
            coordinates.append([float(x), float(y)])
    return np.array(coordinates)

def load_geoid_heights(file_path):
    N = []
    with open(file_path, "r") as file:
        next(file)  # Skip header if present
        next(file)
        for line in file:
            parts = line.split()
            if len(parts) >= 5:
                geoid_height = parts[3]
                ellipsoid_height = parts[4]
                N.append(float(geoid_height) - float(ellipsoid_height))
    return np.array(N)

def load_weights(file_path):
    weights = []
    with open(file_path, "r") as file:
        next(file)  # Skip header if present
        next(file)
        for line in file:
            parts = line.split()
            weight = 1 if parts[5][0] == "T" else 4
            weights.append(weight)
    return np.array(weights)

def calculate_statistics(A, F, weights, X):
    P = np.diag(weights)
    v = (A @ X) - F
    n = len(F)
    e = X.shape[0]

    variance_residual = (v.T @ P @ v) / (n - e)
    covariance_matrix = variance_residual * np.linalg.inv(A.T @ P @ A)
    coefficient_sd = np.sqrt(np.diag(covariance_matrix))

    for i, (coef, sd) in enumerate(zip(X, coefficient_sd)):
        significance_value = abs(coef / sd)
        if significance_value > 2.131:
            print(f"The coefficient {chr(65+i)} is significant with value {significance_value:.3f}")
        else:
            print(f"{chr(65+i)} is NOT significant: {significance_value:.3f}")

    return variance_residual, covariance_matrix, coefficient_sd

def compute_deflection(X, point):
    N_derivative_x = 2*X[0]*point[0] + X[1]*point[1] + X[2]
    N_derivative_y = X[1]*point[0] + X[3]
    deflection = np.array([-N_derivative_x, -N_derivative_y]) * 200000 / np.pi  # Convert to milligrad (milligon)
    return deflection

def N_func(X, x, y):
    # Assuming a polynomial model of degree determined by the length of X
    # Adjust the function according to the actual model used
    return X[0]*x**2 + X[1]*x*y + X[2]*x + X[3]*y + X[4]

def compute_residuals_and_std_deviation(X, coordinates, F, weights):
    # Compute estimated geoid heights for all points
    v_new = np.array([N_func(X, point[0], point[1]) for point in coordinates]) - F
    
    # Compute the P matrix again if needed
    P = np.diag(weights)
    
    # Number of observations and unknowns
    n = len(F)
    e = len(X)
    
    # Compute residuals
    variance_residual_new = (v_new.T @ P @ v_new) / (n - e)
    std_deviation_unit_weight = np.sqrt(variance_residual_new)
    
    print(f"Standard deviation of unit weight: {std_deviation_unit_weight}")
    
    # To compute the standard deviation of levelled heights
    # Assuming levelled heights correspond to the orthometric heights in your dataset
    # Adjust as necessary if your definition differs
    levelled_heights_indices = [i for i, point in enumerate(coordinates) if weights[i] == 4]  # Adjust this condition based on your dataset
    levelled_heights_residuals = v_new[levelled_heights_indices]
    if len(levelled_heights_residuals) > 0:
        std_deviation_levelled_heights = np.sqrt(np.sum(levelled_heights_residuals**2) / len(levelled_heights_residuals))
        print(f"Standard deviation of levelled heights: {std_deviation_levelled_heights}")
    else:
        print("No levelled heights found to compute standard deviation.")
    
    return std_deviation_unit_weight, std_deviation_levelled_heights if 'std_deviation_levelled_heights' in locals() else None

def compute_coefficients(A, F, weights):
    P = np.diag(weights)
    X = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ F
    return X

def main():
    coordinates_file = "/Users/knutlilleaas/Documents/Assignment 2 Teoretisk Geomatikk/xy_coordinates.txt"
    survey_data_file = "/Users/knutlilleaas/Documents/Assignment 2 Teoretisk Geomatikk/survey_data.txt"
    
    coordinates = load_coordinates(coordinates_file)
    N = load_geoid_heights(survey_data_file)
    weights = load_weights(survey_data_file)
    
    A = np.array([[x**2, x*y, y**2, x, y, 1] for x, y in coordinates])
    X = compute_coefficients(A, N, weights)

    print("Coefficients:", X)
    variance_residual, covariance_matrix, coefficient_sd = calculate_statistics(A, N, weights, X)

    print(f"Variance residual: {variance_residual}")
    print(f"Covariance matrix: {covariance_matrix}")
    print(f"Coefficient standard deviations {coefficient_sd}")

    #Reevaluate without C coefficient (insignificant)
    A = np.array([[x**2, x*y, x, y, 1] for x, y in coordinates])
    X = compute_coefficients(A, N, weights)

    print("Coefficients:", X)

    # Task C: Deflection of the vertical
    point1 = coordinates[3]  # MOHOLT
    point2 = coordinates[15]  # SJETNEMARKA

    deflection1 = compute_deflection(X, point1)
    deflection2 = compute_deflection(X, point2)
    print(f"Deflection at point1: {deflection1}, point2: {deflection2}")

    # TASK D:
    compute_residuals_and_std_deviation(X, coordinates, N, weights)


if __name__ == "__main__":
    main()
