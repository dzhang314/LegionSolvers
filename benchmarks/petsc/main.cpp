#include <cstdlib>
#include <cstring>

#include <mpi.h>

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>


enum class MatrixType {
    LAPLACIAN_1D_3POINT,
    LAPLACIAN_2D_5POINT,
    LAPLACIAN_3D_7POINT,
    LAPLACIAN_3D_27POINT,
};


MatrixType get_matrix_type() {
    char dim_name[5];
    PetscBool dim_set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, "-dim", dim_name, 5, &dim_set);
    if (!dim_set) { std::exit(EXIT_FAILURE); }
    if (std::strcmp(dim_name, "1D") == 0) {
        return MatrixType::LAPLACIAN_1D_3POINT;
    } else if (std::strcmp(dim_name, "2D") == 0) {
        return MatrixType::LAPLACIAN_2D_5POINT;
    } else if (std::strcmp(dim_name, "3D") == 0) {
        return MatrixType::LAPLACIAN_3D_7POINT;
    } else if (std::strcmp(dim_name, "3D27") == 0) {
        return MatrixType::LAPLACIAN_3D_27POINT;
    } else {
        std::exit(EXIT_FAILURE);
    }
}


constexpr double two = 2.0;
constexpr double four = 4.0;
constexpr double six = 6.0;
constexpr double neg_one = -1.0;
constexpr double center = 88.0 / 26.0;
constexpr double face = -6.0 / 26.0;
constexpr double edge = -3.0 / 26.0;
constexpr double corner = -2.0 / 26.0;


int main(int argc, char **argv) {

    // Initialize MPI and PETSc.
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

    // Extract matrix type from command line parameters.
    const MatrixType matrix_type = get_matrix_type();

    // Get problem size parameters from command line.
    PetscInt nx = 1;
    PetscInt ny = 1;
    PetscInt nz = 1;
    PetscBool nx_set = PETSC_FALSE;
    PetscBool ny_set = PETSC_FALSE;
    PetscBool nz_set = PETSC_FALSE;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-nx", &nx, &nx_set));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-ny", &ny, &ny_set));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-nz", &nz, &nz_set));

    // Print problem size and version information.
    PetscInt major, minor, subminor, release;
    PetscCall(PetscGetVersionNumber(&major, &minor, &subminor, &release));
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "Running problem of size %d x %d x %d on PETSc version "
        "%d.%d.%d.%d.\n",
        nx,
        ny,
        nz,
        major,
        minor,
        subminor,
        release
    ));

    // Set up distributed matrix data structure.
    Mat A;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    const PetscInt problem_size = nx * ny * nz;
    PetscCall(
        MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, problem_size, problem_size)
    );
    PetscCall(MatSetFromOptions(A));

    // Preallocate storage depending on matrix type.
    switch (matrix_type) {
    case MatrixType::LAPLACIAN_1D_3POINT:
        PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 0, NULL));
        break;
    case MatrixType::LAPLACIAN_2D_5POINT:
        PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 2, NULL));
        break;
    case MatrixType::LAPLACIAN_3D_7POINT:
        PetscCall(MatMPIAIJSetPreallocation(A, 7, NULL, 4, NULL));
        break;
    case MatrixType::LAPLACIAN_3D_27POINT:
        PetscCall(MatMPIAIJSetPreallocation(A, 27, NULL, 24, NULL));
        break;
    }

    // Fill matrix with entries.
    PetscInt row_start, row_end;
    PetscCall(MatGetOwnershipRange(A, &row_start, &row_end));
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    for (PetscInt global_index = row_start; global_index < row_end;
         ++global_index) {
        switch (matrix_type) {
        case MatrixType::LAPLACIAN_1D_3POINT: {
            const PetscInt gx = global_index;
            if (gx > 0) {
                MatSetValue(A, global_index, gx - 1, neg_one, INSERT_VALUES);
            }
            MatSetValue(A, global_index, gx, two, INSERT_VALUES);
            if (gx + 1 < nx) {
                MatSetValue(A, global_index, gx + 1, neg_one, INSERT_VALUES);
            }
        } break;
        case MatrixType::LAPLACIAN_2D_5POINT: {
            const PetscInt gy = global_index % ny;
            const PetscInt gx = global_index / ny;
            if (gx > 0) {
                MatSetValue(
                    A, global_index, ny * (gx - 1) + gy, neg_one, INSERT_VALUES
                );
            }
            if (gy > 0) {
                MatSetValue(
                    A, global_index, ny * gx + (gy - 1), neg_one, INSERT_VALUES
                );
            }
            MatSetValue(A, global_index, ny * gx + gy, four, INSERT_VALUES);
            if (gy + 1 < ny) {
                MatSetValue(
                    A, global_index, ny * gx + (gy + 1), neg_one, INSERT_VALUES
                );
            }
            if (gx + 1 < nx) {
                MatSetValue(
                    A, global_index, ny * (gx + 1) + gy, neg_one, INSERT_VALUES
                );
            }
        } break;
        case MatrixType::LAPLACIAN_3D_7POINT: {
            const PetscInt gz = global_index % nz;
            const PetscInt gxy = global_index / nz;
            const PetscInt gy = gxy % ny;
            const PetscInt gx = gxy / ny;
            if (gx > 0) {
                MatSetValue(
                    A,
                    global_index,
                    ny * nz * (gx - 1) + nz * gy + gz,
                    neg_one,
                    INSERT_VALUES
                );
            }
            if (gy > 0) {
                MatSetValue(
                    A,
                    global_index,
                    ny * nz * gx + nz * (gy - 1) + gz,
                    neg_one,
                    INSERT_VALUES
                );
            }
            if (gz > 0) {
                MatSetValue(
                    A,
                    global_index,
                    ny * nz * gx + nz * gy + (gz - 1),
                    neg_one,
                    INSERT_VALUES
                );
            }
            MatSetValue(
                A, global_index, ny * nz * gx + nz * gy + gz, six, INSERT_VALUES
            );
            if (gz + 1 < nz) {
                MatSetValue(
                    A,
                    global_index,
                    ny * nz * gx + nz * gy + (gz + 1),
                    neg_one,
                    INSERT_VALUES
                );
            }
            if (gy + 1 < ny) {
                MatSetValue(
                    A,
                    global_index,
                    ny * nz * gx + nz * (gy + 1) + gz,
                    neg_one,
                    INSERT_VALUES
                );
            }
            if (gx + 1 < nx) {
                MatSetValue(
                    A,
                    global_index,
                    ny * nz * (gx + 1) + nz * gy + gz,
                    neg_one,
                    INSERT_VALUES
                );
            }
        } break;
        case MatrixType::LAPLACIAN_3D_27POINT: {
            const PetscInt gz = global_index % nz;
            const PetscInt gxy = global_index / nz;
            const PetscInt gy = gxy % ny;
            const PetscInt gx = gxy / ny;
            // clang-format off
            if ((gx > 0) && (gy > 0) && (gz > 0)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * (gy - 1) + (gz - 1), corner, INSERT_VALUES);
            if ((gx > 0) && (gy > 0)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * (gy - 1) + gz, edge, INSERT_VALUES);
            if ((gx > 0) && (gy > 0) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * (gy - 1) + (gz + 1), corner, INSERT_VALUES);
            if ((gx > 0) && (gz > 0)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * gy + (gz - 1), edge, INSERT_VALUES);
            if ((gx > 0)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * gy + gz, face, INSERT_VALUES);
            if ((gx > 0) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * gy + (gz + 1), edge, INSERT_VALUES);
            if ((gx > 0) && (gy + 1 < ny) && (gz > 0)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * (gy + 1) + (gz - 1), corner, INSERT_VALUES);
            if ((gx > 0) && (gy + 1 < ny)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * (gy + 1) + gz, edge, INSERT_VALUES);
            if ((gx > 0) && (gy + 1 < ny) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * (gx - 1) + nz * (gy + 1) + (gz + 1), corner, INSERT_VALUES);
            if ((gy > 0) && (gz > 0)) MatSetValue(A, global_index, ny * nz * gx + nz * (gy - 1) + (gz - 1), edge, INSERT_VALUES);
            if ((gy > 0)) MatSetValue(A, global_index, ny * nz * gx + nz * (gy - 1) + gz, face, INSERT_VALUES);
            if ((gy > 0) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * gx + nz * (gy - 1) + (gz + 1), edge, INSERT_VALUES);
            if ((gz > 0)) MatSetValue(A, global_index, ny * nz * gx + nz * gy + (gz - 1), face, INSERT_VALUES);
            MatSetValue(A, global_index, ny * nz * gx + nz * gy + gz, center, INSERT_VALUES);
            if ((gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * gx + nz * gy + (gz + 1), face, INSERT_VALUES);
            if ((gy + 1 < ny) && (gz > 0)) MatSetValue(A, global_index, ny * nz * gx + nz * (gy + 1) + (gz - 1), edge, INSERT_VALUES);
            if ((gy + 1 < ny)) MatSetValue(A, global_index, ny * nz * gx + nz * (gy + 1) + gz, face, INSERT_VALUES);
            if ((gy + 1 < ny) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * gx + nz * (gy + 1) + (gz + 1), edge, INSERT_VALUES);
            if ((gx + 1 < nx) && (gy > 0) && (gz > 0)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * (gy - 1) + (gz - 1), corner, INSERT_VALUES);
            if ((gx + 1 < nx) && (gy > 0)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * (gy - 1) + gz, edge, INSERT_VALUES);
            if ((gx + 1 < nx) && (gy > 0) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * (gy - 1) + (gz + 1), corner, INSERT_VALUES);
            if ((gx + 1 < nx) && (gz > 0)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * gy + (gz - 1), edge, INSERT_VALUES);
            if ((gx + 1 < nx)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * gy + gz, face, INSERT_VALUES);
            if ((gx + 1 < nx) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * gy + (gz + 1), edge, INSERT_VALUES);
            if ((gx + 1 < nx) && (gy + 1 < ny) && (gz > 0)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * (gy + 1) + (gz - 1), corner, INSERT_VALUES);
            if ((gx + 1 < nx) && (gy + 1 < ny)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * (gy + 1) + gz, edge, INSERT_VALUES);
            if ((gx + 1 < nx) && (gy + 1 < ny) && (gz + 1 < nz)) MatSetValue(A, global_index, ny * nz * (gx + 1) + nz * (gy + 1) + (gz + 1), corner, INSERT_VALUES);
            // clang-format on
        } break;
        }
    }

    // Complete matrix assembly.
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    // PetscViewer viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
    // PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);
    // MatView(A, viewer);

    // Construct a random solution vector.
    Vec x;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, problem_size));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetRandom(x, nullptr));

    // Construct a right-hand-side vector.
    Vec b;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecSetSizes(b, PETSC_DECIDE, problem_size));
    PetscCall(VecSetFromOptions(b));

    // Perform a matrix multiplication to fill the right-hand-side vector.
    PetscCall(MatMult(A, x, b));

    // Clear the solution vector.
    PetscCall(VecZeroEntries(x));

    // Set up Krylov solver data structures.
    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetOperators(ksp, A, A));

    // Run Krylov solver algorithm and time execution.
    PetscLogStage log_stage;
    PetscCall(PetscLogStageRegister("KSPBenchmark", &log_stage));
    PetscLogDouble start, stop;
    PetscCall(PetscTime(&start));
    PetscCall(PetscLogStagePush(log_stage));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(PetscLogStagePop());
    PetscCall(PetscTime(&stop));

    // Print elapsed time per iteration.
    const PetscLogDouble elapsed = stop - start;
    PetscInt num_iterations;
    PetscCall(KSPGetIterationNumber(ksp, &num_iterations));
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "Performed %d iterations.\nTook %f ms per iteration.\n",
        num_iterations,
        1000.0 * elapsed / num_iterations
    ));

    // Finalize MPI and PETSc.
    PetscCall(PetscFinalize());
}
