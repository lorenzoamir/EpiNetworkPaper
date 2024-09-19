# python random_sir.py --outdir output/random_sir --beta 0.3 --mu 0.1 --i0 0.01 --planninghorizon 10 --dt 1 --activitysteps 10 --nsims 100 --nindividuals 1000 --tmax 1000 --seed 42

RANDOM_SIR=1
RANDOM_SIS=0
SCALEFREE_SIR=1
SCALEFREE_SIS=0

# Parameters
alphas=(0.00 0.10 0.13 0.17 0.20 0.28 0.37 0.48 0.63 0.81 1.06 1.37 1.79 2.32 3.02 3.92 5.10 6.63 8.61 11 14 19 25 32 42 54 70 91 120 154 200)
beta=0.3
mu=0.1
i0=0.01
planninghorizon=10
dt=1
activitysteps=10
nsims=100
nindividuals=1000
tmax=1000
seed=42

# Print parameters
echo "alphas: ${alphas[@]}"
echo "beta: $beta"
echo "mu: $mu"
echo "i0: $i0"
echo "planninghorizon: $planninghorizon"
echo "dt: $dt"
echo "activitysteps: $activitysteps"
echo "nsims: $nsims"
echo "nindividuals: $nindividuals"
echo "tmax: $tmax"
echo "seed: $seed"

sleep 3

# Make list of commands
commands=()

if [ $RANDOM_SIR -eq 1 ]; then
    commands+=("random_sir.py")
fi
if [ $RANDOM_SIS -eq 1 ]; then
    commands+=("random_sis.py")
fi
if [ $SCALEFREE_SIR -eq 1 ]; then
    commands+=("scalefree_sir.py")
fi
if [ $SCALEFREE_SIS -eq 1 ]; then
    commands+=("scalefree_sis.py")
fi

for command in "${commands[@]}"; do
    # Strip '.py extension to get output directory
    outputdir="output/${command%.*}"
    echo "Running command: $command"
    echo "Output: output/$(echo $command | cut -d' ' -f2)"

    i=1
    tot=${#alphas[@]}
    for alpha in "${alphas[@]}"; do
        # Print command and wait
        echo "python $command --outdir $outputdir --alpha $alpha --beta $beta --mu $mu --i0 $i0 --planninghorizon $planninghorizon --dt $dt --activitysteps $activitysteps --nsims $nsims --nindividuals $nindividuals --tmax $tmax --seed $seed"
        sleep 1
        # Actually run the command
        python $command --outdir $outputdir --alpha $alpha --beta $beta --mu $mu --i0 $i0 --planninghorizon $planninghorizon --dt $dt --activitysteps $activitysteps --nsims $nsims --nindividuals $nindividuals --tmax $tmax --seed $seed &
        # Print how many simulations have been run
        echo "Simulations run: $i/$tot"
        i=$((i+1))
    done
done


