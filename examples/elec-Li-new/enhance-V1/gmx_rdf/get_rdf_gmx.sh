#!/bin/bash

set -uo pipefail

workdir="/oscar/data/yqi27/xitan25/works/umlp_electrolytes/MLP-MD"
datadir="$workdir/analysis/get_all_rdf_gmx"
tprdir="$workdir/all_tpr"

metals=(Li)
lambda="0.00"

mkdir -p "$datadir"

for metal in "${metals[@]}"; do
    if [[ "$metal" == "Li" ]]; then
        Vs=(1 2 3 4 5 6)
        solvents=(G2 DME FEC EC THF PC)
    else
        Vs=(1 2 3 4 5 6 7)
        solvents=(G2 DME SFL PC FEMC FEC EC)
    fi

    rdf_fsi_dir="$datadir/${metal}-metal/FSI-ML"
    rdf_solvent_dir="$datadir/${metal}-metal/solvent-ML"

    rm -rf "$rdf_fsi_dir" "$rdf_solvent_dir"
    mkdir -p "$rdf_fsi_dir" "$rdf_solvent_dir"

    metal_workdir="$workdir/${metal}-metal"

    for V in "${Vs[@]}"; do
        idx=$((V - 1))
        solvent="${solvents[$idx]}"

        for casedir1 in "$metal_workdir"/case${V}-*; do
            [[ -d "$casedir1" ]] || continue

            casename1=$(basename "$casedir1")

            echo
            echo "================================================================="
            echo "[ SYSTEM START ] $casename1 | Solvent = $solvent"
            echo "================================================================="

            if [[ "$metal" == "Li" ]]; then
                case "$V" in
                    1) subcases=(1 2) ;;
                    2) subcases=(2 3 4 5) ;;
                    3) subcases=(1 2 3 4) ;;
                    4) subcases=(1 2 4 6) ;;
                    5) subcases=(1 2 3) ;;
                    6) subcases=(1 2 3 4) ;;
                    *) subcases=() ;;
                esac
            else
                subcases=(1 2 3 4 5)
            fi

            for sub in "${subcases[@]}"; do
                for casedir2 in "$casedir1"/case${sub}-*; do
                    [[ -d "$casedir2" ]] || continue

                    casename2=$(basename "$casedir2")
                    system_dir="$workdir/${metal}-metal/${casename1}/${casename2}"
                    fep_dir="$system_dir/lambda-$lambda"
                    basic_dir="$system_dir/basicfile"

                    echo
                    echo "[ CASE ] $casename1/$casename2"

                    if [[ ! -d "$fep_dir" ]]; then
                        echo "Missing work dir: $fep_dir"
                        continue
                    fi

                    traj=$(find "$fep_dir" -maxdepth 1 -name "ghost_md.xyz" -type f | head -n 1)
                    if [[ -z "$traj" ]]; then
                        echo "Missing trajectory: $fep_dir/ghost_md.xyz"
                        continue
                    fi

                    src_tpr="$tprdir/${metal}-metal/+1e_tpr/${casename1}_${casename2}/${casename2}.tpr"
                    src_ndx="$tprdir/${metal}-metal/+1e_tpr/${casename1}_${casename2}/${casename2}.index"

                    if [[ ! -f "$src_tpr" ]]; then
                        echo "Missing TPR file: $src_tpr"
                        continue
                    fi

                    if [[ ! -f "$src_ndx" ]]; then
                        echo "Missing index file: $src_ndx"
                        continue
                    fi

                    pdb=$(find "$basic_dir" -maxdepth 1 -name "*.pdb.orig" -type f | head -n 1)
                    if [[ -z "$pdb" ]]; then
                        echo "Missing PDB file in: $basic_dir"
                        continue
                    fi

                    cd "$fep_dir" || continue

                    cp "$src_tpr" nvt.tpr
                    cp "$src_ndx" index.ndx
                    cp "$pdb" top.pdb

                    cp2kxyz2xtc \
                        -itop top.pdb \
                        -itrj "$traj" \
                        -o nvt.xtc \
                        -dt 250 \
                        -format extxyz

                    if [[ ! -f nvt.xtc ]]; then
                        echo "Failed to generate nvt.xtc in $fep_dir"
                        continue
                    fi

                    solvent_out="$rdf_solvent_dir/MLPMD-rdf_${metal}_solvent_${casename1}_${casename2}.xvg"
                    fsi_out="$rdf_fsi_dir/MLPMD-rdf_${metal}_FSI_${casename1}_${casename2}.xvg"

                    echo -e "${metal}\n${solvent}" | gmx rdf \
                        -f nvt.xtc \
                        -s nvt.tpr \
                        -b 100 \
                        -n index.ndx \
                        -bin 0.01 \
                        -o "$solvent_out" \
                        -selrpos mol_com \
                        -seltype mol_com

                    echo -e "${metal}\nFSI" | gmx rdf \
                        -f nvt.xtc \
                        -s nvt.tpr \
                        -b 100 \
                        -n index.ndx \
                        -bin 0.01 \
                        -o "$fsi_out" \
                        -selrpos mol_com \
                        -seltype mol_cog
                done
            done
        done
    done
done