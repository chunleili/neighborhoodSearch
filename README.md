这是一个用太极编写的用于SPH邻域搜索的类。

neighborhood_search.h是类的实现，而main.py是可以运行的测试。

运行main.py将会产生neighbors.txt 和 num_neighbors.txt两个文件。分别代表每个粒子的邻居的编号和邻居的数量。


## 使用方法
1. 建立类的时候需要传入三个参数：
  例如
``` python
ns = NeighborhoodSearchSparse(positions, particle_max_num, 0.04, domain_size)
```
- positions是粒子的位置的三维的taichi field.
- particle_max_num 是粒子的最大数量
- 0.04 是邻域的半径，通常与SPH的核函数的支持半径相同。
- domain_size 是模拟区域的大小，是一个三维的numpy array，例如np.array([10.0, 10.0, 10.0])。

2. 使用run_search方法进行搜索
```python
ns.run_search()
```

3. 搜索结果将保存在neighbors和num_neighbors这两个变量中

- num_neighbors[i]代表第i个粒子的邻居的数量。
- neighbors[i, k]代表第i个粒子的第k个邻居的编号。

## 利用taichi稀疏数据结构节约显存

读者可能已经注意到，目前的类名是NeighborhoodSearchSparse，其中Sparse代表稀疏网格。该特殊的数据结构在网格数量较多的时候会起作用。应当注意：假如网格数量不多，使用稀疏网格反而会增加显存占用。

与正常的taichi field 相比，我们只在grid_particles_num和particles_in_grid这两个变量上做了新的改动。

改动如下

原稠密网格
```python
self.grid_particles_num = ti.field(int, shape=(self.grid_num))
self.particles_in_grid = ti.field(int, shape=(*self.grid_num, self.max_num_particles_in_grid))
```

稀疏网格
```python
self.particles_in_grid = ti.field(int)
self.grid_particles_num = ti.field(int)
self.grid_snode = ti.root.bitmasked(ti.ijk, self.grid_num)
self.grid_snode.place(self.grid_particles_num)
self.grid_snode.bitmasked(ti.l, self.max_num_particles_in_grid).place(self.particles_in_grid)
```

我们只需要在__init__函数中替换上面的代码即可使用稀疏网格。

另外，为了展示网格到底有多稀疏（实际被占用的比例），我增加了一个简单的函数grid_usage。该函数只是遍历所有网格，然后统计。注意：必须使用struct-for 遍历才能保持稀疏性，range for会破坏稀疏性。


那么，内存的节约效果到底有多好呢？我们可以做一个简单的测试进行对比。我们通过改动ti.init(arch=ti.gpu, device_memory_GB=xxx)来进行测试。显存不足时会发生报错。以此来测试显存占用。

main.py中的test_small对应的是区域大小为1.0的小区域，这时候我们会发现0.15GB的显存足以应对稠密网格，而不足以应对稀疏网格。这是因为稀疏数据结构带来了额外的开销。这时候使用稀疏数据结构反而是得不偿失的。

main.py中的test_large对应的是区域大小为10.0的大区域，这时候我们会发现3.3GB的显存足以应对稀疏网格，而不足以应对稠密网格。这时候就体现了稀疏网格的优越性。

最后我们需要指出：该程序中内存占用最多的变量实际上就是网格相关的变量。因为三维情况下网格的数量通常远超粒子的数量。所以我们才仅仅考虑优化网格相关的变量，而不必考虑其他变量的优化。

## 邻域搜索的原理

邻域搜索是借助网格来进行空间搜索的。它的原理有些类似桶排序（但它不排序）。核心哲学就是一句话：一个粒子的邻居仅可能在其相邻的空间区域内。

因此，我们首先找到粒子所在的网格的3x3的stencil，然后在该stencil内逐个比较距离即可。

为此，我们先要知道每个粒子在哪一个网格之中。这个容易，只需要用粒子的位置整除网格的大小即可。为了后续的使用，我们必须要有这样一个变量：particles_in_grid。顾名思义，particles_in_grid代表某个网格里有哪些粒子。例如particles_in_grid[grid_index, k]即第grid_index个网格里的第k个粒子的编号。当然，我们还需要一个变量记忆网格内有几个粒子，这就不再赘述。这里grid_index可以使用三维下标，也可以使用展开后的一维下标。本程序里我们使用的是三维下标。上述过程在update_grid函数中实现。

必须要特别注意的是：由于我们在编写并行的程序，所以必须要考虑**读写冲突**的问题。多个粒子可能会同时写入到同一个网格当中。所以下面这种写法几乎是唯一正确的写法。
```
k = ti.atomic_add(self.grid_particles_num[grid_index], 1)
self.particles_in_grid[grid_index, k] = i
```

然后，我们进行真正的搜索，将搜索结果保存在neighbors和num_neighbors这两个变量中。这个过程在store_neighbors函数中实现。这里同样要注意读写冲突的问题，多个粒子可能会同时写入到同一个粒子的邻居当中。所以下面这种写法几乎是唯一正确的写法。
```
kk = ti.atomic_add(self.num_neighbors[p_i], 1)
self.neighbors[p_i, kk] = p_j
```

还需要额外注意的一点是使用stencil时必须考虑超出边界的问题。例如在边界0处的网格的stencil会出现网格下标为-1。is_in_grid函数就是为了防止超出边界的。

另外，在执行store_neighbors和update_grid之前，注意要清空已经使用了的数据结构。

上面所述的所有过程，实际上都是在run_search函数中完成的。


读者若有兴趣，还可以思考借助排序来进行邻域搜索。这样有两点好处：

1）无需particles_in_grid等占用大量显存的变量，因为排序后的粒子本身就是按照空间位置顺序来的。甚至无需存储neighbors这个变量。
2）由于排序后的数组是按照空间位置连续的，而使用邻域搜索的大多数操作都是基于近邻的。所以对于内存读写速度有一定的提升，因而性能上有所增益。

但是，这样做也是有代价的。由于排序会打乱原数组的顺序，所以后续假如有基于数组顺序的操作，就会变得麻烦。或许可以考虑额外存储particle_id来记忆原粒子顺序，并随之排序，然后在需要粒子原下标时取出particle_id。