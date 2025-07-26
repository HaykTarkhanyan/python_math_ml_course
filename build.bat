@echo off
echo Building all math homework solutions...

echo Building Calculus homework...
cd math\Homeworks\solutions_quarto
quarto render 03_calculus.qmd --profile solution
cd ..\..\..

echo Building Vectors homework...
cd math\Homeworks\solutions_quarto
quarto render 01_vectors.qmd --profile solution
cd ..\..\..

echo Building Matrices homework...
cd math\Homeworks\solutions_quarto
quarto render 02_matrices.qmd --profile solution
cd ..\..\..

echo Building Several Variables ^& Probability homework...
cd math\Homeworks\solutions_quarto
quarto render 04_several_variables_probability.qmd --profile solution
cd ..\..\..

echo Building Random Variables homework...
cd math\Homeworks\solutions_quarto
quarto render 05_random_variables.qmd --profile solution
cd ..\..\..

echo All math homeworks built successfully!
pause
