#pragma once

#include "utils/exception.h"
#include "utils/mathUtils.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>


namespace OptiSik {

/// Simplex solver for linear programming problems
///	Handles both integer and non-integer variables
///	Handles all types of constraints (<=, >=, =)
/// Internally uses the original simplex algorithm and thus might
/// not be efficient for large matrices.
template <typename T, typename TInteger = int64_t, typename TTolerance = Tolerance<T>>
class SimplexSolver {
    using IntegerOperations      = IntegerOperations<T, TInteger>;
    static constexpr T tolerance = TTolerance::tolerance;

public:
    /// Specifies the type of constraint
    enum class ConstraintType { LEQUAL = 0, GEQUAL = 1, EQUAL = 2 };

    /// Defines a constraint
    ///	In a form of coefs * x (type) value
    ///	Where coefs is a vector of coefficients for each variable, type is the
    /// constraint type (<=, >=, =), and value is the constant value
    struct Constraint {
        std::vector<T> coefs;
        ConstraintType type;
        T value;
    };

    /// Specifies the bounds of an optimized variable
    enum class VariableBounds {
        NON_NEGATIVE = 0,
        NON_POSITIVE = 1,
        UNBOUNDED    = 2
    };

    /// Specifies whether the variable is integer or float
    enum class VariableType { FLOAT = 0, INTEGER = 1 };

private:
    /// Defines an optimized variable
    struct Variable {
        /// Cost coefficient in the cost function
        T cost;

        /// Bounds type
        VariableBounds bounds;

        /// Whether the variable is integer or float
        VariableType type;

        /// If this variable was added during conversion to standard form, this
        /// points to the original variable Otherwise std::nullopt
        std::optional<size_t> originalIndex;
    };

    /// List of optimized variables
    /// Can contain additional variables added during conversion to standard
    /// form
    std::vector<Variable> mVariables;

    /// List of constraints
    using Constraints = std::vector<Constraint>;
    Constraints mConstraints;

    /// Number of original variables
    const size_t mVariableCount;

public:
    /// Creates a simplex solver with the given number of variables
    /// Initializes all variables to be float, have cost 1 and non-negative
    /// bounds
    SimplexSolver(const size_t variableCount)
    : mVariables(variableCount,
                 { .cost          = T(1),
                   .bounds        = VariableBounds::NON_NEGATIVE,
                   .type          = VariableType::FLOAT,
                   .originalIndex = -1 }),
      mVariableCount(variableCount) {
        if (variableCount == 0) {
            throw invalidArgument(
            "SimplexSolver: Must have at least one variable");
        }
    }

    /// Adds a constraint to the solver. The constraint must have the same
    /// number of coefficients as the number of variables
    void addConstraint(const Constraint& constraint) {
        if (constraint.coefs.size() != mVariables.size()) {
            throw invalidArgument(
            "SimplexSolver::addConstraint: Coefficient size mismatch");
        }
        mConstraints.push_back(constraint);
    }

    /// Sets the properties of a variable at the given index
    void setVariable(const size_t index,
                     const T cost,
                     const VariableBounds bounds = VariableBounds::NON_NEGATIVE,
                     const VariableType type     = VariableType::FLOAT) {
        if (index >= mVariables.size()) {
            throw invalidArgument(
            "SimplexSolver::setVariable: Index out of range");
        }
        mVariables[index].cost   = cost;
        mVariables[index].bounds = bounds;
        mVariables[index].type   = type;
    }

    /// Prints the current state of the solver (variables, cost function,
    /// constraints)
    void print() const {
        std::cout << "Variables" << std::endl;
        for (size_t i = 0; i < mVariables.size(); ++i) {
            printVariable(i);
            std::cout << std::endl;
        }
        printCostFunction();
        std::cout << std::endl;

        std::cout << "Constraints" << std::endl;
        for (const Constraint& c : mConstraints) {
            printConstraint(c);
            std::cout << std::endl;
        }
    }

    /// Converts the problem to standard form that we can solve with the simplex
    /// algorithm
    void convertToStandardForm() {
        // Find all variables that are not non-negative
        std::vector<size_t> newVars;
        for (size_t i = 0; i < mVariables.size(); ++i) {
            switch (mVariables[i].bounds) {
            case VariableBounds::NON_NEGATIVE: break;
            case VariableBounds::UNBOUNDED: newVars.push_back(i); break;
            case VariableBounds::NON_POSITIVE: {
                newVars.push_back(i);
                // Add new constraint ensuring the variable is <= 0
                Constraint c;
                c.value = T(0);
                c.type  = ConstraintType::LEQUAL;
                c.coefs.resize(mVariables.size(), 0);
                c.coefs[i] = T(1);
                mConstraints.push_back(c);
                break;
            }
            }
            mVariables[i].bounds = VariableBounds::NON_NEGATIVE;
        }
        // Add new variables needed to convert to non-negative
        for (size_t originalIndex : newVars) {
            Variable v;
            v.cost          = -mVariables[originalIndex].cost;
            v.type          = mVariables[originalIndex].type;
            v.bounds        = VariableBounds::NON_NEGATIVE;
            v.originalIndex = originalIndex;
            mVariables.push_back(v);
            // Add new variable to constraints
            for (Constraint& c : mConstraints) {
                c.coefs.push_back(-c.coefs[originalIndex]);
            }
        }
        // Fix constraints to have positive constant
        for (Constraint& c : mConstraints) {
            if (c.value < T(0)) {
                c.value = -c.value;
                for (auto& coef : c.coefs) {
                    coef = -coef;
                }
                if (c.type == ConstraintType::LEQUAL) {
                    c.type = ConstraintType::GEQUAL;
                } else if (c.type == ConstraintType::GEQUAL) {
                    c.type = ConstraintType::LEQUAL;
                }
            }
        }
    }

    /// Type of optimization to perform
    enum class Operation { MINIMIZE = 0, MAXIMIZE = 1 };

    /// Result of the simplex algorithm execution
    struct Result {
        /// Optimal values for each variable
        std::vector<T> variables;

        /// Optimal cost value
        T cost;

        void print() const {
            for (size_t i = 0; i < variables.size(); ++i) {
                std::cout << char('a' + i) << " = " << variables[i] << std::endl;
            }
            std::cout << "Cost = " << cost << std::endl;
        }

        bool empty() const {
            return variables.empty();
        }

        bool isBetter(const Result& other, const Operation operation) const {
            return !empty() &&
            (other.empty() || (operation == Operation::MINIMIZE && cost < other.cost) ||
             (operation == Operation::MAXIMIZE && cost > other.cost));
        }
    };

    /// Executes the simplex algorithm to minimize or maximize the cost function
    Result execute(const Operation operation) const {
        SimplexSolver solver = *this;
        solver.convertToStandardForm();
        const Result floatResult = solver.executeFloat(operation);
        if (floatResult.empty() || !hasAnyIntegerVariables()) {
            return solver.convertResult(floatResult);
        }

        // We need to handle integer variables - perform branch and bound
        std::vector<Constraints> stack;
        Result bestResult;

        const auto branchAndBound = [&](const Result& result) {
            if (!result.isBetter(bestResult, operation)) {
                return;
            }
            if (const auto index = solver.findTypeViolation(result); index != std::nullopt) {
                const T bound =
                IntegerOperations::removeFraction(result.variables[*index]);
                stack.push_back(addLowerBound(solver.mConstraints, *index, bound + T(1)));
                stack.push_back(addUpperBound(solver.mConstraints, *index, bound));
            } else {
                bestResult = result;
            }
        };

        // Start branching by taking the original floating-point result
        branchAndBound(floatResult);

        while (!stack.empty()) {
            solver.mConstraints = std::move(stack.back());
            stack.pop_back();
            solver.convertToStandardForm();
            branchAndBound(solver.executeFloat(operation));
        }
        return solver.convertResult(bestResult);
    }

    /// Converts the result obtained from the internal optimization to the
    /// result matching the original variables
    Result convertResult(const Result result) const {
        if (result.empty()) {
            return result;
        }
        assert(result.variables.size() == mVariables.size());
        Result newResult;
        newResult.cost = result.cost;
        newResult.variables.resize(mVariableCount, 0);
        for (size_t i = 0; i < mVariableCount; ++i) {
            newResult.variables[i] = result.variables[i];
        }
        for (size_t i = mVariableCount; i < mVariables.size(); ++i) {
            const Variable& v = mVariables[i];
            assert(v.originalIndex.has_value());
            newResult.variables[*v.originalIndex] -= result.variables[i];
        }
        return newResult;
    }

    /// Checks whether the result is valid for the current problem
    template <const bool TPrint = true>
    bool checkResult(const Result result) const {
        if (result.empty()) {
            if (TPrint) {
                std::cout << "Result is empty - no solution was found" << std::endl;
            }
            return true;
        }
        assert(result.variables.size() == mVariables.size());
        // Check constraints
        bool ok = true;
        for (const Constraint& c : mConstraints) {
            T value = T(0);
            for (size_t i = 0; i < c.coefs.size(); ++i) {
                value += c.coefs[i] * result.variables[i];
            }
            bool cOk = true;
            switch (c.type) {
            case ConstraintType::LEQUAL:
                cOk &= (value <= c.value + tolerance);
                break;
            case ConstraintType::GEQUAL:
                cOk &= (value >= c.value - tolerance);
                break;
            case ConstraintType::EQUAL:
                cOk &= (abs(value - c.value) <= tolerance);
                break;
            }
            if (!cOk) {
                ok = false;
                if (TPrint) {
                    std::cout << "Constraint violated: ";
                    printConstraint(c);
                    std::cout << " (computed value " << value << ", expected "
                              << c.value << ")" << std::endl;
                }
            }
        }
        // Check variable type and bounds
        T computedCost = T(0);
        for (size_t i = 0; i < mVariables.size(); ++i) {
            const Variable& v = mVariables[i];
            computedCost += v.cost * result.variables[i];
            bool vOk = true;
            switch (v.bounds) {
            case VariableBounds::NON_NEGATIVE:
                vOk &= (result.variables[i] >= -tolerance);
                break;
            case VariableBounds::NON_POSITIVE:
                vOk &= (result.variables[i] <= +tolerance);
                break;
            case VariableBounds::UNBOUNDED: break;
            }
            if (!vOk) {
                ok = false;
                if (TPrint) {
                    std::cout << "Variable ";
                    printVariable(i);
                    std::cout << " out of bounds: value " << result.variables[i]
                              << std::endl;
                }
            }
            if (v.type == VariableType::INTEGER &&
                integerVariableError(result.variables[i]) > tolerance) {
                ok = false;
                if (TPrint) {
                    std::cout << "Variable ";
                    printVariable(i);
                    std::cout << " out of bounds: value " << result.variables[i]
                              << std::endl;
                }
            }
        }
        // Check cost
        if (abs(computedCost - result.cost) > tolerance) {
            ok = false;
            if (TPrint) {
                std::cout << "Computed cost " << computedCost << " does not match reported cost "
                          << result.cost << std::endl;
            }
        }
        return ok;
    }

private:
    /// Equation used internally during optimization
    /// constant = coefs * x
    struct Equation {
        /// Coefficients for each variable
        std::vector<T> coefs;

        /// Constant value
        T constant;

        /// Current value of the basis variable associated with this equation
        T basisValue;

        /// Index of the basis variable associated with this equation
        size_t basis;

        /// Whether the basis variable is artificial
        bool isBasisArtifical;

        /// Subtracts a scaled version of another equation from this equation to
        /// eliminate a variable
        void subtractScaledEquation(const Equation& eq, const size_t pivot) {
            const T multiplier = coefs[pivot];
            if (multiplier == T(0)) {
                return;
            }
            for (size_t i = 0; i < coefs.size(); ++i) {
                coefs[i] -= multiplier * eq.coefs[i];
            }
            constant -= multiplier * eq.constant;
            // This should be already true, but we set it to 0 to avoid
            // numerical issues
            coefs[pivot] = T(0);
        }

        /// Moves the pivot variable into the basis of this equation
        void movePivotToBasis(const size_t pivot, const T newBasisValue) {
            // Move pivot variable into basis
            const T div = coefs[pivot];
            for (T& c : coefs) {
                c /= div;
            }
            constant /= div;
            basis            = pivot;
            basisValue       = newBasisValue;
            isBasisArtifical = false;
            coefs[pivot]     = T(1); // To avoid numerical issues
        }
    };

    /// Internal optimization execution - disregards variable types
    /// (integer/float)
    Result executeFloat(const Operation operation) const {
        const size_t variableCount = computeVariableCount();
        size_t artificialCount     = 0;
        std::vector<Equation> equations = createEquations(variableCount, artificialCount);
        Equation costFunction = createCostEquation(equations, operation);

        const size_t nonArtificialCount = variableCount - artificialCount;
        std::optional<size_t> pivot = findPivot(equations, costFunction, nonArtificialCount);
        while (pivot.has_value()) {
            // Find minimum pivot variable increase
            T minimum                            = T(0);
            std::optional<size_t> minimumEqIndex = std::nullopt;
            for (size_t i = 0; i < equations.size(); ++i) {
                const Equation& e = equations[i];
                if (e.coefs[*pivot] <= T(0)) {
                    continue;
                }
                const T pivotValue = e.constant / e.coefs[*pivot];
                if (!minimumEqIndex.has_value() || minimum > pivotValue) {
                    minimum        = pivotValue;
                    minimumEqIndex = i;
                }
            }

            if (!minimumEqIndex.has_value()) {
                // Unbounded solution
                throw computationError("SimplexSolver: Unbounded solution");
            }

            // Move pivot variable into basis
            equations[*minimumEqIndex].movePivotToBasis(*pivot, costFunction.coefs[*pivot]);

            // Update remaining equations
            for (size_t i = 0; i < equations.size(); ++i) {
                if (i != *minimumEqIndex) {
                    equations[i].subtractScaledEquation(equations[*minimumEqIndex], *pivot);
                }
            }

            pivot = findPivot(equations, costFunction, nonArtificialCount);
        }

        // Report the found optimal solution
        return reportResult(equations);
    }

    /// Computes the total number of variables tracked during the actual
    /// optimization
    size_t computeVariableCount() const {
        size_t variableCount = mVariables.size();
        for (const auto& c : mConstraints) {
            if (c.type != ConstraintType::GEQUAL) {
                // EQUAL = 1 artificial
                // LEQUAL = 1 slack
                ++variableCount;
            } else {
                // GEQUAL = 1 artificial + 1 surplass
                variableCount += 2;
            }
        }
        return variableCount;
    }

    /// Creates equations from the given constraints for the optimization
    /// process
    std::vector<Equation>
    createEquations(const size_t variableCount, size_t& artificialCount) const {
        std::vector<Equation> equations;
        size_t newVarIndex       = mVariables.size();
        size_t newArtificalIndex = variableCount - 1;
        for (size_t i = 0; i < mConstraints.size(); ++i) {
            Equation e;
            e.coefs.resize(variableCount, 0);
            for (size_t j = 0; j < mConstraints[i].coefs.size(); ++j) {
                e.coefs[j] = mConstraints[i].coefs[j];
            }
            e.constant = mConstraints[i].value;
            if (mConstraints[i].type == ConstraintType::LEQUAL) {
                e.coefs[newVarIndex] = T(1); // 1 slack
                e.basis              = newVarIndex;
                e.isBasisArtifical   = false;
                e.basisValue         = T(0);
                ++newVarIndex;
            } else if (mConstraints[i].type == ConstraintType::GEQUAL) {
                e.coefs[newVarIndex]       = -T(1); // 1 surplas
                e.coefs[newArtificalIndex] = T(1);  // 1 artificial
                e.basis                    = newArtificalIndex;
                e.isBasisArtifical         = true;
                e.basisValue               = -T(1);
                ++newVarIndex;
                --newArtificalIndex;
            } else {
                e.coefs[newArtificalIndex] = T(1); // 1 artificial
                e.basis                    = newArtificalIndex;
                e.isBasisArtifical         = true;
                e.basisValue               = -T(1);
                --newArtificalIndex;
            }
            equations.emplace_back(e);
        }
        artificialCount = variableCount - newArtificalIndex - 1;
        assert(newVarIndex == newArtificalIndex + 1);
        return equations;
    }

    /// Creates the cost equation used during optimization
    Equation createCostEquation(const std::vector<Equation>& equations,
                                const Operation operation) const {
        Equation costFunction;
        costFunction.constant = T(0);
        costFunction.coefs.resize(equations[0].coefs.size(), 0);
        const T mult = operation == Operation::MINIMIZE ? -T(1) : T(1);
        for (size_t j = 0; j < mVariables.size(); ++j) {
            costFunction.coefs[j] = mult * mVariables[j].cost;
        }
        return costFunction;
    }

    /// Finds the pivot variable to enter the basis
    /// Returns std::nullopt if no such variable exists (optimal solution found)
    std::optional<size_t> findPivot(const std::vector<Equation>& equations,
                                    const Equation& costFunction,
                                    const size_t nonArtificialCount) const {
        std::optional<size_t> pivot = std::nullopt;
        // We track values containing artificial variables separately, since
        // they should have higher value then other variables
        T prevValue = 0;
        T prevArtificialValue(0);
        for (size_t i = 0; i < nonArtificialCount; ++i) {
            // Compute cost function minus current basis contribution
            T value = costFunction.coefs[i];
            T artificialValue(0);
            for (const Equation& eq : equations) {
                if (eq.isBasisArtifical) {
                    artificialValue -= eq.basisValue * eq.coefs[i];
                } else {
                    value -= eq.basisValue * eq.coefs[i];
                }
            }
            if (prevArtificialValue + tolerance < artificialValue ||
                (abs(prevArtificialValue - artificialValue) < tolerance &&
                 prevValue + tolerance < value)) {
                prevValue           = value;
                prevArtificialValue = artificialValue;
                pivot               = i;
            }
        }
        return pivot;
    }

    /// Reports the result of the optimization
    Result reportResult(const std::vector<Equation>& equations) const {
        std::vector<T> basisValues(mVariables.size(), 0);
        for (const Equation& eq : equations) {
            /// Check for infeasible solution due to artificial basis variable
            /// having positive value
            if (eq.isBasisArtifical && eq.constant > tolerance) {
                return Result{ .variables = {}, .cost = T(0) }; // Infeasible solution
            }
        }
        Result r;
        r.variables.resize(mVariables.size(), 0);
        r.cost = T(0);
        for (const Equation& eq : equations) {
            if (eq.basis < mVariables.size()) {
                r.variables[eq.basis] = eq.constant;
                r.cost += eq.constant * mVariables[eq.basis].cost;
            }
        }
        return r;
    }

    /// Prints one variable
    void printVariable(const size_t i) const {
        switch (mVariables[i].type) {
        case VariableType::FLOAT: std::cout << "float "; break;
        case VariableType::INTEGER: std::cout << "int "; break;
        }
        std::cout << char('a' + i);
        switch (mVariables[i].bounds) {
        case VariableBounds::NON_NEGATIVE: std::cout << " >= 0 "; break;
        case VariableBounds::UNBOUNDED: break;
        case VariableBounds::NON_POSITIVE: std::cout << " <= 0 "; break;
        }
        if (mVariables[i].originalIndex.has_value()) {
            std::cout << " (added to " << char('a' + *mVariables[i].originalIndex) << ')';
        }
    }

    /// Prints the cost function
    void printCostFunction() const {
        std::cout << "Cost function = ";
        for (size_t i = 0; i < mVariables.size(); ++i) {
            if (mVariables[i].cost != T(0)) {
                if (mVariables[i].cost > T(0) && i > 0) {
                    std::cout << "+";
                }
                std::cout << mVariables[i].cost << char('a' + i) << " ";
            }
        }
    }

    /// Prints one constraint
    void printConstraint(const Constraint& c) const {
        for (size_t j = 0; j < c.coefs.size(); ++j) {
            if (c.coefs[j] != T(0)) {
                if (c.coefs[j] > T(0) && j > 0) {
                    std::cout << "+";
                }
                std::cout << c.coefs[j] << char('a' + j) << " ";
            }
        }
        switch (c.type) {
        case ConstraintType::LEQUAL: std::cout << " <= " << c.value; break;
        case ConstraintType::GEQUAL: std::cout << " >= " << c.value; break;
        case ConstraintType::EQUAL: std::cout << " = " << c.value;
        }
    }

    /// Computes the distance of a given floating-point value from the closest
    /// integer value
    static T integerVariableError(const T floatValue) {
        const T value = IntegerOperations::removeFraction(floatValue);
        const T difference =
        std::min(abs(floatValue - value), abs(floatValue - value - 1));
        return difference;
    }

    /// Finds the variable that violates its type the most (integer variables
    /// having non-integer values)
    std::optional<size_t> findTypeViolation(const Result& result) const {
        T maxDifference             = T(0);
        std::optional<size_t> index = std::nullopt;
        for (size_t i = 0; i < mVariables.size(); ++i) {
            if (mVariables[i].type == VariableType::INTEGER) {
                const T difference = integerVariableError(result.variables[i]);
                if (difference > maxDifference) {
                    maxDifference = difference;
                    index         = i;
                }
            }
        }
        return maxDifference < tolerance ? std::nullopt : index;
    }

    /// Checks whether there are any integer variables in the problem
    bool hasAnyIntegerVariables() const {
        for (const auto& var : mVariables) {
            if (var.type == VariableType::INTEGER) {
                return true;
            }
        }
        return false;
    }

    /// Adds a lower bound constraint for the given variable to the list of
    /// constraints
    std::vector<Constraint> addLowerBound(const std::vector<Constraint>& constraints,
                                          const size_t varIndex,
                                          const T boundValue) const {
        return addBound(constraints, varIndex, boundValue, ConstraintType::GEQUAL);
    }

    /// Adds an upper bound constraint for the given variable to the list of
    /// constraints
    std::vector<Constraint> addUpperBound(const std::vector<Constraint>& constraints,
                                          const size_t varIndex,
                                          const T boundValue) const {
        return addBound(constraints, varIndex, boundValue, ConstraintType::LEQUAL);
    }

    /// Adds a bound constraint for the given variable to the list of
    /// constraints
    std::vector<Constraint> addBound(const std::vector<Constraint>& constraints,
                                     const size_t varIndex,
                                     const T boundValue,
                                     const ConstraintType boundType) const {
        std::vector<Constraint> newConstraints = constraints;
        Constraint c;
        c.coefs.resize(mVariables.size(), 0);
        c.coefs[varIndex] = T(1);
        c.type            = boundType;
        c.value           = boundValue;
        newConstraints.push_back(c);
        return newConstraints;
    }
};

} // namespace OptiSik