#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Version.hpp>

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

using Scalar = double;
using LocalIndex = Tpetra::Map<>::local_ordinal_type;
using GlobalIndex = Tpetra::Map<>::global_ordinal_type;
using NodeType =
    Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Cuda, Kokkos::CudaUVMSpace>;

using Map = Tpetra::Map<LocalIndex, GlobalIndex, NodeType>;
using MultiVector =
    Tpetra::MultiVector<Scalar, LocalIndex, GlobalIndex, NodeType>;
using Operator = Tpetra::Operator<Scalar, LocalIndex, GlobalIndex, NodeType>;
using CSRMatrix = Tpetra::CrsMatrix<Scalar, LocalIndex, GlobalIndex, NodeType>;


std::vector<std::string> get_command_line_args(int argc, char **argv) {
    std::vector<std::string> result;
    for (int i = 0; i < argc; ++i) { result.emplace_back(argv[i]); }
    return result;
}


int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard scope(&argc, &argv);
    {
        Tpetra::global_size_t nx = 1;
        Tpetra::global_size_t ny = 1;
        Tpetra::global_size_t nz = 1;

        const auto args = get_command_line_args(argc, argv);
        if (args[1] == "1D") {
            nx = static_cast<Tpetra::global_size_t>(std::stoull(args[3]));
        } else if (args[1] == "2D") {
            nx = static_cast<Tpetra::global_size_t>(std::stoull(args[3]));
            ny = static_cast<Tpetra::global_size_t>(std::stoull(args[4]));
        } else if ((args[1] == "3D") || (args[1] == "3D27")) {
            nx = static_cast<Tpetra::global_size_t>(std::stoull(args[3]));
            ny = static_cast<Tpetra::global_size_t>(std::stoull(args[4]));
            nz = static_cast<Tpetra::global_size_t>(std::stoull(args[5]));
        } else {
            std::cerr << "INVALID ARGUMENTS" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        const Tpetra::global_size_t total_size = nx * ny * nz;

        auto comm = Tpetra::getDefaultComm();
        const int rank = comm->getRank();
        if (rank == 0) {
            std::cout << "Running problem of size " << nx << " x " << ny
                      << " x " << nz << " on " << comm->getSize()
                      << " processors." << std::endl;
            std::cout << Tpetra::version() << std::endl << std::endl;
        }

        const GlobalIndex index_origin = 0;

        Teuchos::RCP<const Map> map =
            Teuchos::rcp(new Map(total_size, index_origin, comm));

        const size_t local_elements = map->getLocalNumElements();

        Teuchos::RCP<CSRMatrix> A;
        constexpr Scalar two = static_cast<Scalar>(2.0);
        constexpr Scalar four = static_cast<Scalar>(4.0);
        constexpr Scalar six = static_cast<Scalar>(6.0);
        constexpr Scalar neg_one = static_cast<Scalar>(-1.0);

        constexpr Scalar center =
            static_cast<Scalar>(88.0) / static_cast<Scalar>(26.0);
        constexpr Scalar face =
            static_cast<Scalar>(-6.0) / static_cast<Scalar>(26.0);
        constexpr Scalar edge =
            static_cast<Scalar>(-3.0) / static_cast<Scalar>(26.0);
        constexpr Scalar corner =
            static_cast<Scalar>(-2.0) / static_cast<Scalar>(26.0);

        std::vector<GlobalIndex> indices;
        std::vector<Scalar> values;

        if (args[1] == "1D") {
            A = Teuchos::rcp(new CSRMatrix(map, 3));
            indices.reserve(3);
            values.reserve(3);
            for (LocalIndex local_index = 0;
                 local_index < static_cast<LocalIndex>(local_elements);
                 ++local_index) {
                const GlobalIndex global_index =
                    map->getGlobalElement(local_index);
                const GlobalIndex gx = global_index;
                if (gx > 0) {
                    indices.push_back(gx - 1);
                    values.push_back(neg_one);
                }
                indices.push_back(gx);
                values.push_back(two);
                if (gx + 1 < nx) {
                    indices.push_back(gx + 1);
                    values.push_back(neg_one);
                }
                A->insertGlobalValues(global_index, indices, values);
                indices.clear();
                values.clear();
            }
        } else if (args[1] == "2D") {
            A = Teuchos::rcp(new CSRMatrix(map, 5));
            indices.reserve(5);
            values.reserve(5);
            for (LocalIndex local_index = 0;
                 local_index < static_cast<LocalIndex>(local_elements);
                 ++local_index) {
                const GlobalIndex global_index =
                    map->getGlobalElement(local_index);
                const GlobalIndex gy = global_index % ny;
                const GlobalIndex gx = global_index / ny;
                if (gx > 0) {
                    indices.push_back(ny * (gx - 1) + gy);
                    values.push_back(neg_one);
                }
                if (gy > 0) {
                    indices.push_back(ny * gx + (gy - 1));
                    values.push_back(neg_one);
                }
                indices.push_back(ny * gx + gy);
                values.push_back(four);
                if (gy + 1 < ny) {
                    indices.push_back(ny * gx + (gy + 1));
                    values.push_back(neg_one);
                }
                if (gx + 1 < nx) {
                    indices.push_back(ny * (gx + 1) + gy);
                    values.push_back(neg_one);
                }
                A->insertGlobalValues(global_index, indices, values);
                indices.clear();
                values.clear();
            }
        } else if (args[1] == "3D") {
            A = Teuchos::rcp(new CSRMatrix(map, 7));
            indices.reserve(7);
            values.reserve(7);
            for (LocalIndex local_index = 0;
                 local_index < static_cast<LocalIndex>(local_elements);
                 ++local_index) {
                const GlobalIndex global_index =
                    map->getGlobalElement(local_index);
                const GlobalIndex gz = global_index % nz;
                const GlobalIndex gxy = global_index / nz;
                const GlobalIndex gy = gxy % ny;
                const GlobalIndex gx = gxy / ny;
                if (gx > 0) {
                    indices.push_back(ny * nz * (gx - 1) + nz * gy + gz);
                    values.push_back(neg_one);
                }
                if (gy > 0) {
                    indices.push_back(ny * nz * gx + nz * (gy - 1) + gz);
                    values.push_back(neg_one);
                }
                if (gz > 0) {
                    indices.push_back(ny * nz * gx + nz * gy + (gz - 1));
                    values.push_back(neg_one);
                }
                indices.push_back(ny * nz * gx + nz * gy + gz);
                values.push_back(six);
                if (gz + 1 < nz) {
                    indices.push_back(ny * nz * gx + nz * gy + (gz + 1));
                    values.push_back(neg_one);
                }
                if (gy + 1 < ny) {
                    indices.push_back(ny * nz * gx + nz * (gy + 1) + gz);
                    values.push_back(neg_one);
                }
                if (gx + 1 < nx) {
                    indices.push_back(ny * nz * (gx + 1) + nz * gy + gz);
                    values.push_back(neg_one);
                }
                A->insertGlobalValues(global_index, indices, values);
                indices.clear();
                values.clear();
            }
        } else if (args[1] == "3D27") {
            A = Teuchos::rcp(new CSRMatrix(map, 27));
            indices.reserve(27);
            values.reserve(27);
            for (LocalIndex local_index = 0;
                 local_index < static_cast<LocalIndex>(local_elements);
                 ++local_index) {
                const GlobalIndex global_index =
                    map->getGlobalElement(local_index);
                const GlobalIndex gz = global_index % nz;
                const GlobalIndex gxy = global_index / nz;
                const GlobalIndex gy = gxy % ny;
                const GlobalIndex gx = gxy / ny;
                if ((gx > 0) && (gy > 0) && (gz > 0)) {
                    indices.push_back(
                        ny * nz * (gx - 1) + nz * (gy - 1) + (gz - 1)
                    );
                    values.push_back(corner);
                }
                if ((gx > 0) && (gy > 0)) {
                    indices.push_back(ny * nz * (gx - 1) + nz * (gy - 1) + gz);
                    values.push_back(edge);
                }
                if ((gx > 0) && (gy > 0) && (gz + 1 < nz)) {
                    indices.push_back(
                        ny * nz * (gx - 1) + nz * (gy - 1) + (gz + 1)
                    );
                    values.push_back(corner);
                }
                if ((gx > 0) && (gz > 0)) {
                    indices.push_back(ny * nz * (gx - 1) + nz * gy + (gz - 1));
                    values.push_back(edge);
                }
                if ((gx > 0)) {
                    indices.push_back(ny * nz * (gx - 1) + nz * gy + gz);
                    values.push_back(face);
                }
                if ((gx > 0) && (gz + 1 < nz)) {
                    indices.push_back(ny * nz * (gx - 1) + nz * gy + (gz + 1));
                    values.push_back(edge);
                }
                if ((gx > 0) && (gy + 1 < ny) && (gz > 0)) {
                    indices.push_back(
                        ny * nz * (gx - 1) + nz * (gy + 1) + (gz - 1)
                    );
                    values.push_back(corner);
                }
                if ((gx > 0) && (gy + 1 < ny)) {
                    indices.push_back(ny * nz * (gx - 1) + nz * (gy + 1) + gz);
                    values.push_back(edge);
                }
                if ((gx > 0) && (gy + 1 < ny) && (gz + 1 < nz)) {
                    indices.push_back(
                        ny * nz * (gx - 1) + nz * (gy + 1) + (gz + 1)
                    );
                    values.push_back(corner);
                }
                if ((gy > 0) && (gz > 0)) {
                    indices.push_back(ny * nz * gx + nz * (gy - 1) + (gz - 1));
                    values.push_back(edge);
                }
                if ((gy > 0)) {
                    indices.push_back(ny * nz * gx + nz * (gy - 1) + gz);
                    values.push_back(face);
                }
                if ((gy > 0) && (gz + 1 < nz)) {
                    indices.push_back(ny * nz * gx + nz * (gy - 1) + (gz + 1));
                    values.push_back(edge);
                }
                if ((gz > 0)) {
                    indices.push_back(ny * nz * gx + nz * gy + (gz - 1));
                    values.push_back(face);
                }
                indices.push_back(ny * nz * gx + nz * gy + gz);
                values.push_back(center);
                if ((gz + 1 < nz)) {
                    indices.push_back(ny * nz * gx + nz * gy + (gz + 1));
                    values.push_back(face);
                }
                if ((gy + 1 < ny) && (gz > 0)) {
                    indices.push_back(ny * nz * gx + nz * (gy + 1) + (gz - 1));
                    values.push_back(edge);
                }
                if ((gy + 1 < ny)) {
                    indices.push_back(ny * nz * gx + nz * (gy + 1) + gz);
                    values.push_back(face);
                }
                if ((gy + 1 < ny) && (gz + 1 < nz)) {
                    indices.push_back(ny * nz * gx + nz * (gy + 1) + (gz + 1));
                    values.push_back(edge);
                }
                if ((gx + 1 < nx) && (gy > 0) && (gz > 0)) {
                    indices.push_back(
                        ny * nz * (gx + 1) + nz * (gy - 1) + (gz - 1)
                    );
                    values.push_back(corner);
                }
                if ((gx + 1 < nx) && (gy > 0)) {
                    indices.push_back(ny * nz * (gx + 1) + nz * (gy - 1) + gz);
                    values.push_back(edge);
                }
                if ((gx + 1 < nx) && (gy > 0) && (gz + 1 < nz)) {
                    indices.push_back(
                        ny * nz * (gx + 1) + nz * (gy - 1) + (gz + 1)
                    );
                    values.push_back(corner);
                }
                if ((gx + 1 < nx) && (gz > 0)) {
                    indices.push_back(ny * nz * (gx + 1) + nz * gy + (gz - 1));
                    values.push_back(edge);
                }
                if ((gx + 1 < nx)) {
                    indices.push_back(ny * nz * (gx + 1) + nz * gy + gz);
                    values.push_back(face);
                }
                if ((gx + 1 < nx) && (gz + 1 < nz)) {
                    indices.push_back(ny * nz * (gx + 1) + nz * gy + (gz + 1));
                    values.push_back(edge);
                }
                if ((gx + 1 < nx) && (gy + 1 < ny) && (gz > 0)) {
                    indices.push_back(
                        ny * nz * (gx + 1) + nz * (gy + 1) + (gz - 1)
                    );
                    values.push_back(corner);
                }
                if ((gx + 1 < nx) && (gy + 1 < ny)) {
                    indices.push_back(ny * nz * (gx + 1) + nz * (gy + 1) + gz);
                    values.push_back(edge);
                }
                if ((gx + 1 < nx) && (gy + 1 < ny) && (gz + 1 < nz)) {
                    indices.push_back(
                        ny * nz * (gx + 1) + nz * (gy + 1) + (gz + 1)
                    );
                    values.push_back(corner);
                }
                A->insertGlobalValues(global_index, indices, values);
                indices.clear();
                values.clear();
            }
        } else {
            std::cerr << "INVALID ARGUMENTS" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        A->fillComplete();
        A->print(std::cout);

        Teuchos::RCP<MultiVector> x = Teuchos::rcp(new MultiVector(map, 1));
        Teuchos::RCP<MultiVector> b = Teuchos::rcp(new MultiVector(map, 1));
        x->randomize();
        A->apply(*x, *b);
        x->scale(0.0);

        Belos::LinearProblem<Scalar, MultiVector, Operator> Problem(A, x, b);
        if (!Problem.setProblem()) {
            std::cout
                << "ERROR: Belos::LinearProblem failed to set up correctly !"
                << std::endl;
            return EXIT_FAILURE;
        }

        Teuchos::RCP<Belos::SolverManager<Scalar, MultiVector, Operator>>
            solver;

        if (args[2] == "CG") {
            Teuchos::ParameterList BelosList;
            BelosList.set("Block Size", 1);
            BelosList.set("Maximum Iterations", 200);
            BelosList.set("Convergence Tolerance", 0.0);
            solver = Teuchos::rcp(
                new Belos::BlockCGSolMgr<Scalar, MultiVector, Operator>(
                    Teuchos::rcp(&Problem, false),
                    Teuchos::rcp(&BelosList, false)
                )
            );
        } else if (args[2] == "BiCGStab") {
            Teuchos::ParameterList BelosList;
            BelosList.set("Maximum Iterations", 200);
            BelosList.set("Convergence Tolerance", 0.0);
            solver = Teuchos::rcp(
                new Belos::BiCGStabSolMgr<Scalar, MultiVector, Operator>(
                    Teuchos::rcp(&Problem, false),
                    Teuchos::rcp(&BelosList, false)
                )
            );
        } else if (args[2] == "GMRES") {
            Teuchos::ParameterList BelosList;
            BelosList.set("Block Size", 1);
            BelosList.set("Adaptive Block Size", false);
            BelosList.set("Num Blocks", 10);
            BelosList.set("Maximum Iterations", 200);
            BelosList.set("Maximum Restarts", 20);
            BelosList.set("Convergence Tolerance", 0.0);
            solver = Teuchos::rcp(
                new Belos::BlockGmresSolMgr<Scalar, MultiVector, Operator>(
                    Teuchos::rcp(&Problem, false),
                    Teuchos::rcp(&BelosList, false)
                )
            );
        } else {
            std::cerr << "INVALID ARGUMENTS" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        const auto begin = std::chrono::high_resolution_clock::now();
        Belos::ReturnType ret = solver->solve();
        const auto end = std::chrono::high_resolution_clock::now();

        const auto elapsed_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count()
        );

        const int numIters = solver->getNumIters();

        std::cout << "Performed " << numIters << " iterations." << std::endl;
        std::cout << "Took " << (elapsed_ns / numIters / 1.0e6)
                  << " ms per iteration." << std::endl;
    }
    return EXIT_SUCCESS;
}
