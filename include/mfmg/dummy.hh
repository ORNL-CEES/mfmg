#include <deal.II/grid/tria.h>

class Dummy
{
  public:
    Dummy() = default;

    void generate_mesh(dealii::Triangulation<2> &tria);
};
