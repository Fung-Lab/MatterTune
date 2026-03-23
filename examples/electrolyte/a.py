from ase.io import read, write

atoms_list = read("./outputs/md/md.xyz", index=":")
write("md.xyz", atoms_list)