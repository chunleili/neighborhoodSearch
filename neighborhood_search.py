import taichi as ti
import numpy as np


@ti.data_oriented
class NeighborhoodSearchSparse:
    def __init__(self,positions,particle_max_num, support_radius, domain_size, use_sparse_grid=False):
        self.particle_max_num = particle_max_num
        self.support_radius = support_radius
        self.domain_size = domain_size

        self.positions = positions

        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.grid_num_1d = self.grid_num[0] * self.grid_num[1] * self.grid_num[2]
        self.dim = 3
        self.max_num_neighbors = 60
        self.max_num_particles_in_grid = 50

        self.neighbors = ti.field(int, shape=(self.particle_max_num, self.max_num_neighbors))
        self.num_neighbors = ti.field(int, shape=self.particle_max_num)

        if not use_sparse_grid:
            self.grid_particles_num = ti.field(int, shape=(self.grid_num))
            self.particles_in_grid = ti.field(int, shape=(*self.grid_num, self.max_num_particles_in_grid))
        else:
            self.particles_in_grid = ti.field(int)
            self.grid_particles_num = ti.field(int)
            self.grid_snode = ti.root.bitmasked(ti.ijk, self.grid_num)
            self.grid_snode.place(self.grid_particles_num)
            self.grid_snode.bitmasked(ti.l, self.max_num_particles_in_grid).place(self.particles_in_grid)

    @ti.kernel
    def grid_usage(self) -> ti.f32:
        cnt = 0
        for I in ti.grouped(self.grid_snode):
            if ti.is_active(self.grid_snode, I):
                cnt += 1
        usage = cnt / (self.grid_num_1d)
        return usage

    def deactivate_grid(self):
        self.grid_snode.deactivate_all()

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.kernel
    def update_grid(self):
        for i in range(self.particle_max_num):
            grid_index = self.pos_to_index(self.positions[i])
            k = ti.atomic_add(self.grid_particles_num[grid_index], 1)
            self.particles_in_grid[grid_index, k] = i

    def run_search(self):
        self.num_neighbors.fill(0)
        self.neighbors.fill(-1)
        self.particles_in_grid.fill(-1)
        self.grid_particles_num.fill(0)

        self.update_grid()
        self.store_neighbors()
        # print("Grid usage: ", self.grid_usage())

    @ti.func
    def is_in_grid(self, c):
        return 0 <= c[0] < self.grid_num[0] and 0 <= c[1] < self.grid_num[1] and 0 <= c[2] < self.grid_num[2]

    @ti.kernel
    def store_neighbors(self):
        for p_i in range(self.particle_max_num):
            center_cell = self.pos_to_index(self.positions[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                grid_index = center_cell + offset
                if self.is_in_grid(grid_index):
                    for k in range(self.grid_particles_num[grid_index]):
                        p_j = self.particles_in_grid[grid_index, k]
                        if p_i != p_j and (self.positions[p_i] - self.positions[p_j]).norm() < self.support_radius:
                            kk = ti.atomic_add(self.num_neighbors[p_i], 1)
                            self.neighbors[p_i, kk] = p_j
