#include <mfmg/exceptions.hpp>

#include <boost/program_options.hpp>

#include "test_hierarchy_helpers.hpp"

template <int dim>
void main_(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<dim>;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;

  auto laplace_ptree = params->get_child("laplace");
  Laplace<dim, DVector> laplace(comm, 1);
  laplace.setup_system(laplace_ptree);
  laplace.assemble_system(source, *material_property);

  auto const &a = laplace._system_matrix;
  auto const locally_owned_dofs = laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(laplace._system_rhs);

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
    solution[index] = distribution(generator);

  std::shared_ptr<MeshEvaluator> evaluator(new TestMeshEvaluator<dim>(
      laplace._dof_handler, laplace._constraints, a, material_property));
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params);

  pcout << "Grid complexity    : " << hierarchy.grid_complexity() << std::endl;
  pcout << "Operator complexity: " << hierarchy.operator_complexity()
        << std::endl;

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  a.vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  // pcout << "#0: " << 1.0 << std::endl;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.apply(rhs, solution);

    a.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    double rel_residual = residual.l2_norm() / residual0_norm;
    // pcout << "#" << i + 1 << ": " << rel_residual << std::endl;
    res[i + 1] = rel_residual;
  }

  double const conv_rate = res[n_cycles] / res[n_cycles - 1];
  pcout << "Convergence rate: " << std::fixed << std::setprecision(2)
        << conv_rate << std::endl;
}

int main(int argc, char *argv[])
{
  namespace boost_po = boost::program_options;

  using f_type = std::vector<std::string>;

  MPI_Init(&argc, &argv);

  boost_po::options_description cmd("Available options");
  cmd.add_options()("filename,f", boost_po::value<f_type>()->multitoken(),
                    "file(s) containing input parameters");
  cmd.add_options()("dim,d", boost_po::value<int>(), "dimension");

  boost_po::variables_map vm;
  boost_po::store(boost_po::parse_command_line(argc, argv, cmd), vm);
  boost_po::notify(vm);

  f_type filenames = {"hierarchy_input.info"};
  if (vm.count("filename"))
    filenames = vm["filename"].as<f_type>();

  int dim = 2;
  if (vm.count("dim"))
    dim = vm["dim"].as<int>();

  mfmg::ASSERT(dim == 2 || dim == 3, "Dimension must be 2 or 3");

  for (auto &filename : filenames)
  {
    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info(filename, *params);

    if (dim == 2)
      main_<2>(params);
    else
      main_<3>(params);
  }

  MPI_Finalize();

  return 0;
}
