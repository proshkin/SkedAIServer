#ifndef SAT_RUNNER_HEADER
#define SAT_RUNNER_HEADER

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/flags.h"
#include "absl/log/initialize.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ortools/algorithms/sparse_permutation.h"
#include "ortools/base/helpers.h"
#include "ortools/base/options.h"
#include "ortools/base/timer.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"
#include "ortools/sat/boolean_problem.h"
#include "ortools/sat/boolean_problem.pb.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/lp_utils.h"
#include "ortools/sat/model.h"
#include "ortools/sat/opb_reader.h"
#include "ortools/sat/optimization.h"
#include "ortools/sat/pb_constraint.h"
#include "ortools/sat/sat_base.h"
#include "ortools/sat/sat_cnf_reader.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/sat/sat_solver.h"
#include "ortools/sat/simplification.h"
#include "ortools/sat/symmetry.h"
#include "ortools/util/file_util.h"
#include "ortools/util/logging.h"
#include "ortools/util/strong_integers.h"
#include "ortools/util/time_limit.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#endif



ABSL_FLAG(
    std::string, input, "",
    "Required: input file of the problem to solve. Many format are supported:"
    ".cnf (sat, max-sat, weighted max-sat), .opb (pseudo-boolean sat/optim) "
    "and by default the LinearBooleanProblem proto (binary or text).");

ABSL_FLAG(
    std::string, output, "",
    "If non-empty, write the input problem as a LinearBooleanProblem proto to "
    "this file. By default it uses the binary format except if the file "
    "extension is '.txt'. If the problem is SAT, a satisfiable assignment is "
    "also written to the file.");

ABSL_FLAG(std::string, callback, "",
      "If true, take the input file name and use the callback method to "
      "output the variables every change.");

ABSL_FLAG(int, step_wait, 1000,
      "Alters the initial wait before starting to print solutions "
      "if the callback flag is used.");

ABSL_FLAG(int, print_wait, 1000,
      "Alters the wait in between printing solutions after the "
      "initial solution is printed.");

ABSL_FLAG(int, objective_print_wait, 1000,
      "Alters the wait between which the current objective is printed.");

ABSL_FLAG(std::string, params, "",
          "Parameters for the sat solver in a text format of the "
          "SatParameters proto, example: --params=use_conflicts:true.");

ABSL_FLAG(bool, strict_validity, false,
          "If true, stop if the given input is invalid (duplicate literals, "
          "out of range, zero cofficients, etc.)");

ABSL_FLAG(
    std::string, lower_bound, "",
    "If not empty, look for a solution with an objective value >= this bound.");

ABSL_FLAG(
    std::string, upper_bound, "",
    "If not empty, look for a solution with an objective value <= this bound.");

ABSL_FLAG(int, randomize, 500,
          "If positive, solve that many times the problem with a random "
          "decision heuristic before trying to optimize it.");

ABSL_FLAG(bool, use_symmetry, false,
          "If true, find and exploit the eventual symmetries "
          "of the problem.");

ABSL_FLAG(bool, presolve, true,
          "Only work on pure SAT problem. If true, presolve the problem.");

ABSL_FLAG(bool, probing, false, "If true, presolve the problem using probing.");

ABSL_FLAG(bool, use_cp_model, true,
          "Whether to interpret everything as a CpModelProto or "
          "to read by default a CpModelProto.");

ABSL_FLAG(bool, reduce_memory_usage, false,
          "If true, do not keep a copy of the original problem in memory."
          "This reduce the memory usage, but disable the solution cheking at "
          "the end.");

namespace operations_research {
namespace sat {
namespace {

auto startTime = std::chrono::steady_clock::now();
std::mutex mtx;
std::condition_variable cv;
bool hasNewObjective = false;
bool hasNewSolution = false;
bool hadNewSolution = false;
bool allSolutionsFound = false;
int lastStatus = -1;
int lastObjective = -2000000000;
std::vector<int> lastSolution;
std::chrono::time_point<std::chrono::steady_clock> lastPrintTime;



bool printThreadStarted = false;
std::atomic<bool> changedInLastSecond{true};
std::atomic<bool> modelSolved{false};
std::atomic<bool> variablesChanged{true};
std::atomic<int> stepWait(1);
std::atomic<int> printWait(5);
std::atomic<int> objectivePrintWait(5);

// runs in a separate thread to print the variables every 5 seconds if they have changed
void printVariables(std::atomic<int>& stepWait, std::atomic<int>& printWait, std::atomic<int>& objectivePrintWait, const std::vector<int>& indexes, const CpSolverResponse& r, std::atomic<bool>& changed, std::atomic<bool>& isSolved, std::atomic<bool>& variablesChanged) {
    auto lastOutputTime = std::chrono::steady_clock::now();
    auto lastChangeTime = std::chrono::steady_clock::now();
    auto lastObjectiveTime = std::chrono::steady_clock::now();
    bool firstTime = true;
    int previousObjective = r.objective_value();
    std::cout << "OBJECTIVE: " << r.objective_value() * 1000;

    while (!isSolved) { // Loop until isSolved becomes true
        auto currentTime = std::chrono::steady_clock::now();
        auto printTimeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastOutputTime).count();
        auto changeTimeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastChangeTime).count();
        auto objectiveTimeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastObjectiveTime).count();
        if (variablesChanged.load()) {
            variablesChanged.store(false);
            lastChangeTime = currentTime;
        }
        if (objectiveTimeDiff > objectivePrintWait) {
            if (r.objective_value() != previousObjective) {
                std::cout << r.objective_value();
                lastObjectiveTime = currentTime;
            }
        }
        if (changeTimeDiff >= stepWait && (firstTime || printTimeDiff >= printWait)) {
            variablesChanged.store(false);
            firstTime = false;
            std::cout << r.status() << "," << r.objective_value();
            for (int index : indexes) {
                if (index >= 0 && index < r.solution_size()) {
                    std::cout << "," << r.solution(index);
                }
            }
            std::cout << std::endl;
            lastOutputTime = currentTime;
        }
    }
    std::cout << r.status() << "," << r.objective_value();
    for (int index : indexes) {
        if (index >= 0 && index < r.solution_size()) {
            std::cout << "," << r.solution(index);
        }
    }
    std::cout << std::endl;
}

// Returns a trivial best bound. The best bound corresponds to the lower bound
// (resp. upper bound) in case of a minimization (resp. maximization) problem.
double GetScaledTrivialBestBound(const LinearBooleanProblem& problem) {
  Coefficient best_bound(0);
  const LinearObjective& objective = problem.objective();
  for (const int64_t value : objective.coefficients()) {
    if (value < 0) best_bound += Coefficient(value);
  }
  return AddOffsetAndScaleObjectiveValue(problem, best_bound);
}

bool LoadBooleanProblem(const std::string& filename,
                        LinearBooleanProblem* problem, CpModelProto* cp_model) {
  if (absl::EndsWith(filename, ".opb") ||
      absl::EndsWith(filename, ".opb.bz2")) {
    OpbReader reader;
    if (!reader.Load(filename, problem)) {
      LOG(FATAL) << "Cannot load file '" << filename << "'.";
    }
  } else if (absl::EndsWith(filename, ".cnf") ||
             absl::EndsWith(filename, ".cnf.gz") ||
             absl::EndsWith(filename, ".wcnf") ||
             absl::EndsWith(filename, ".wcnf.gz")) {
    SatCnfReader reader;
    if (absl::GetFlag(FLAGS_use_cp_model)) {
      if (!reader.Load(filename, cp_model)) {
        LOG(FATAL) << "Cannot load file '" << filename << "'.";
      }
    } else {
      if (!reader.Load(filename, problem)) {
        LOG(FATAL) << "Cannot load file '" << filename << "'.";
      }
    }
  } else if (absl::GetFlag(FLAGS_use_cp_model)) {
    LOG(INFO) << "Reading a CpModelProto.";
    *cp_model = ReadFileToProtoOrDie<CpModelProto>(filename);
  } else {
    LOG(INFO) << "Reading a LinearBooleanProblem.";
    *problem = ReadFileToProtoOrDie<LinearBooleanProblem>(filename);
  }
  return true;
}

std::string SolutionString(const LinearBooleanProblem& problem,
                           const std::vector<bool>& assignment) {
  std::string output;
  BooleanVariable limit(problem.original_num_variables());
  for (BooleanVariable index(0); index < limit; ++index) {
    if (index > 0) output += " ";
    absl::StrAppend(&output,
                    Literal(index, assignment[index.value()]).SignedValue());
  }
  return output;
}


void printObjective(const std::chrono::milliseconds& period) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);

        if (!hasNewObjective && !allSolutionsFound) {
            cv.wait(lock, [] { return hasNewObjective || allSolutionsFound; });
        }

        if (allSolutionsFound) {
            break;  // All solutions found and printed, exit the loop.
        }

        auto targetTime = lastPrintTime + period;
        while (hasNewObjective && std::chrono::steady_clock::now() < targetTime && !allSolutionsFound) {
            cv.wait_until(lock, targetTime);  // Wait until the period expires
        }

        if (allSolutionsFound) {
            break;  // All solutions found and printed, exit the loop.
        }

        if (hasNewObjective) {  // Print the latest solution found within the period
            auto timeFromStart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
            std::cout << timeFromStart << "," << lastStatus << "," << lastObjective << std::endl;
            lastPrintTime = std::chrono::steady_clock::now();
            hasNewObjective = false;
        }

    }
}

void printCurrentSolution() {
    auto timeFromStart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cout << timeFromStart << "," << lastStatus << "," << lastObjective;
    for (int solution : lastSolution) {
        std::cout << "," << solution;
    }
    std::cout << std::endl;
}

void printSolution(const std::chrono::milliseconds& period) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);

        if (cv.wait_for(lock, period, [] { return hasNewSolution || allSolutionsFound; })) {
            if (allSolutionsFound) {
                if (hadNewSolution || hasNewSolution || hasNewObjective) {
                    printCurrentSolution();
                }
                break;
            }
            // If a new solution is found within period, reset the timer.
            hasNewSolution = false;
            hadNewSolution = true;
        }
        else {
            // If period elapsed without a new solution
            if (hadNewSolution) {
                printCurrentSolution();
                hasNewObjective = false;
                hadNewSolution = false;
            }
        }

    }
}

void copy_solution(const CpSolverResponse& r, std::vector<int>& indexes, bool isAllSolutionsFound) {
    int objective = r.objective_value();
    int status = r.status();

    std::unique_lock<std::mutex> lock(mtx);
    allSolutionsFound = isAllSolutionsFound;

    if (objective > lastObjective || status != lastStatus) {

        lastObjective = objective;
        lastStatus = status;

        for (int i = 0; i < indexes.size(); i++) {
            lastSolution[i] = r.solution(indexes[i]);
        }

        hasNewSolution = true;
        hasNewObjective = true;
        cv.notify_all();
    }
    else if (allSolutionsFound) {
        // hadNewSolution = false;
        cv.notify_all();
    }
}

// To benefit from the operations_research namespace, we put all the main() code
// here.
int Run() {
  startTime = std::chrono::steady_clock::now();

  SatParameters parameters;
  if (absl::GetFlag(FLAGS_input).empty()) {
    LOG(FATAL) << "Please supply a data file with --input=";
  }

  // Parse the --params flag.
  // parameters.set_log_search_progress(true);
  if (!absl::GetFlag(FLAGS_params).empty()) {
    CHECK(google::protobuf::TextFormat::MergeFromString(
        absl::GetFlag(FLAGS_params), &parameters))
        << absl::GetFlag(FLAGS_params);
  }

  // Initialize the solver.
  std::unique_ptr<SatSolver> solver(new SatSolver());
  solver->SetParameters(parameters);

  // Read the problem.
  LinearBooleanProblem problem;
  CpModelProto cp_model;

  if (!LoadBooleanProblem(absl::GetFlag(FLAGS_input), &problem, &cp_model)) {
    CpSolverResponse response;
    response.set_status(CpSolverStatus::MODEL_INVALID);
    return EXIT_SUCCESS;
  }
  if (!absl::GetFlag(FLAGS_use_cp_model)) {
    LOG(INFO) << "Converting to CpModelProto ...";
    cp_model = BooleanProblemToCpModelproto(problem);
  }

  if (absl::GetFlag(FLAGS_use_cp_model)) {
        problem.Clear();  // We no longer need it, release memory.
        Model model;
        model.Add(NewSatParameters(parameters));

        std::vector<int> indexes;
        bool threadStarted = false;

        stepWait.store(absl::GetFlag(FLAGS_step_wait));
        printWait.store(absl::GetFlag(FLAGS_print_wait));
        objectivePrintWait.store(absl::GetFlag(FLAGS_objective_print_wait));

        std::chrono::milliseconds objectivePeriod(objectivePrintWait);  // e.g., 500 milliseconds
        std::chrono::milliseconds solutionPeriod(printWait);  // e.g., 500 milliseconds

        lastPrintTime = std::chrono::steady_clock::now() - objectivePeriod;  // Initialize to allow immediate print

        std::thread objectivePrinter(printObjective, objectivePeriod);
        std::thread solutionPrinter(printSolution, solutionPeriod);



        if (!absl::GetFlag(FLAGS_callback).empty()) {

            std::string filename = absl::GetFlag(FLAGS_callback);

            std::ifstream file(filename);
            std::string line;

            if (file.is_open()) {
                if (std::getline(file, line)) {
                    std::stringstream ss(line);
                    std::string value;

                    while (std::getline(ss, value, ',')) {
                        indexes.push_back(std::stoi(value));
                    }
                }
                file.close();
            }
            else {
                std::cerr << "Unable to open file\n";
            }
        }

        lastSolution.resize(indexes.size());

        model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
            copy_solution(r, indexes, false);
            

        /*
            if (!printThreadStarted) {
                printThreadStarted = true;
                std::thread printThread(printVariables, std::ref(stepWait), std::ref(printWait), std::ref(objectivePrintWait), indexes, r, std::ref(changedInLastSecond), std::ref(modelSolved), std::ref(variablesChanged));
                printThread.detach();
            }
            changedInLastSecond = true;
            variablesChanged.store(true);
            */
        }));

        const CpSolverResponse response = SolveCpModel(cp_model, &model);
        copy_solution(response, indexes, true);

        objectivePrinter.join();
        solutionPrinter.join();

        if (!absl::GetFlag(FLAGS_output).empty()) {
          if (absl::EndsWith(absl::GetFlag(FLAGS_output), "txt")) {
            CHECK_OK(file::SetTextProto(absl::GetFlag(FLAGS_output), response,
                                        file::Defaults()));
          } else {
            CHECK_OK(file::SetBinaryProto(absl::GetFlag(FLAGS_output), response,
                                          file::Defaults()));
          }
        }

        modelSolved = true;

        // The SAT competition requires a particular exit code and since we don't
        // really use it for any other purpose, we comply.
        if (response.status() == CpSolverStatus::OPTIMAL) return 10;
        if (response.status() == CpSolverStatus::FEASIBLE) return 10;
        if (response.status() == CpSolverStatus::INFEASIBLE) return 20;
        return EXIT_SUCCESS;
  }

  if (absl::GetFlag(FLAGS_strict_validity)) {
    const absl::Status status = ValidateBooleanProblem(problem);
    if (!status.ok()) {
      LOG(ERROR) << "Invalid Boolean problem: " << status.message();
      return EXIT_FAILURE;
    }
  }

  // Count the time from there.
  WallTimer wall_timer;
  UserTimer user_timer;
  wall_timer.Start();
  user_timer.Start();
  double scaled_best_bound = GetScaledTrivialBestBound(problem);

  // Probing.
  SatPostsolver probing_postsolver(problem.num_variables());
  LinearBooleanProblem original_problem;
  if (absl::GetFlag(FLAGS_probing)) {
    // TODO(user): This is nice for testing, but consumes memory.
    original_problem = problem;
    ProbeAndSimplifyProblem(&probing_postsolver, &problem);
  }

  // Load the problem into the solver.
  if (absl::GetFlag(FLAGS_reduce_memory_usage)) {
    if (!LoadAndConsumeBooleanProblem(&problem, solver.get())) {
      LOG(INFO) << "UNSAT when loading the problem.";
    }
  } else {
    if (!LoadBooleanProblem(problem, solver.get())) {
      LOG(INFO) << "UNSAT when loading the problem.";
    }
  }
  auto strtoint64 = [](const std::string& word) {
    int64_t value = 0;
    if (!word.empty()) CHECK(absl::SimpleAtoi(word, &value));
    return value;
  };
  if (!AddObjectiveConstraint(
          problem, !absl::GetFlag(FLAGS_lower_bound).empty(),
          Coefficient(strtoint64(absl::GetFlag(FLAGS_lower_bound))),
          !absl::GetFlag(FLAGS_upper_bound).empty(),
          Coefficient(strtoint64(absl::GetFlag(FLAGS_upper_bound))),
          solver.get())) {
    LOG(INFO) << "UNSAT when setting the objective constraint.";
  }

  // Symmetries!
  //
  // TODO(user): To make this compatible with presolve, we just need to run
  // it after the presolve step.
  if (absl::GetFlag(FLAGS_use_symmetry)) {
    CHECK(!absl::GetFlag(FLAGS_reduce_memory_usage)) << "incompatible";
    CHECK(!absl::GetFlag(FLAGS_presolve)) << "incompatible";
    LOG(INFO) << "Finding symmetries of the problem.";
    std::vector<std::unique_ptr<SparsePermutation>> generators;
    FindLinearBooleanProblemSymmetries(problem, &generators);
    std::unique_ptr<SymmetryPropagator> propagator(new SymmetryPropagator);
    for (int i = 0; i < generators.size(); ++i) {
      propagator->AddSymmetry(std::move(generators[i]));
    }
    solver->AddPropagator(propagator.get());
    solver->TakePropagatorOwnership(std::move(propagator));
  }

  // Optimize?
  std::vector<bool> solution;
  SatSolver::Status result = SatSolver::LIMIT_REACHED;
  parameters.set_log_search_progress(true);
  solver->SetParameters(parameters);
  if (absl::GetFlag(FLAGS_presolve)) {
    std::unique_ptr<TimeLimit> time_limit =
        TimeLimit::FromParameters(parameters);
    SolverLogger logger;
    result = SolveWithPresolve(&solver, time_limit.get(), &solution,
                               /*drat_proof_handler=*/nullptr, &logger);
    if (result == SatSolver::FEASIBLE) {
      CHECK(IsAssignmentValid(problem, solution));
    }
  } else {
    result = solver->Solve();
    if (result == SatSolver::FEASIBLE) {
      ExtractAssignment(problem, *solver, &solution);
      CHECK(IsAssignmentValid(problem, solution));
    }
  }

  // Print the solution status.
  if (result == SatSolver::FEASIBLE) {
    absl::PrintF("s SATISFIABLE\n");

    // Check and output the solution.
    CHECK(IsAssignmentValid(problem, solution));
    if (!absl::GetFlag(FLAGS_output).empty()) {
      CHECK(!absl::GetFlag(FLAGS_reduce_memory_usage)) << "incompatible";
      if (result == SatSolver::FEASIBLE) {
        StoreAssignment(solver->Assignment(), problem.mutable_assignment());
      }
      if (absl::EndsWith(absl::GetFlag(FLAGS_output), ".txt")) {
        CHECK_OK(file::SetTextProto(absl::GetFlag(FLAGS_output), problem,
                                    file::Defaults()));
      } else {
        CHECK_OK(file::SetBinaryProto(absl::GetFlag(FLAGS_output), problem,
                                      file::Defaults()));
      }
    }
  }
  if (result == SatSolver::INFEASIBLE) {
    absl::PrintF("s UNSATISFIABLE\n");
  }

  // Print status.
  // absl::PrintF("c status: %s\n", SatStatusString(result));

  // Print objective value.
  if (solution.empty()) {
    absl::PrintF("c objective: na\n");
    absl::PrintF("c best bound: na\n");
  } else {
    const Coefficient objective = ComputeObjectiveValue(problem, solution);
    absl::PrintF("c objective: %.16g\n",
                 AddOffsetAndScaleObjectiveValue(problem, objective));
    absl::PrintF("c best bound: %.16g\n", scaled_best_bound);
  }

  // Print final statistics.
  absl::PrintF("c booleans: %d\n", solver->NumVariables());
  absl::PrintF("c conflicts: %d\n", solver->num_failures());
  absl::PrintF("c branches: %d\n", solver->num_branches());
  absl::PrintF("c propagations: %d\n", solver->num_propagations());
  absl::PrintF("c walltime: %f\n", wall_timer.Get());
  absl::PrintF("c usertime: %f\n", user_timer.Get());
  absl::PrintF("c deterministic_time: %f\n", solver->deterministic_time());

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace sat
}  // namespace operations_research

static const char kUsage[] =
    "Usage: see flags.\n"
    "This program solves a given problem with the CP-SAT solver.";

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetProgramUsageMessage(kUsage);
  absl::ParseCommandLine(argc, argv);
  return operations_research::sat::Run();
}