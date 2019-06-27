/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#ifndef MFMG_MATERIAL_PROPERTY_HPP
#define MFMG_MATERIAL_PROPERTY_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/vectorization.h>

template <int dim>
class Source final : public dealii::Function<dim>
{
public:
  Source() = default;

  virtual ~Source() override = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override
  {
    return 0.;
  }
};

template <int dim>
class Coefficient : public dealii::Function<dim>
{
public:
  virtual ~Coefficient() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const = 0;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const component = 0) const override = 0;
};

template <int dim>
class ConstantMaterialProperty final : public Coefficient<dim>
{
public:
  ConstantMaterialProperty() = default;

  virtual ~ConstantMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &,
        unsigned int const = 0) const override
  {
    return dealii::make_vectorized_array<double>(1.);
  }

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override
  {
    return 1.;
  }
};

template <int dim>
class LinearXMaterialProperty final : public Coefficient<dim>
{
public:
  LinearXMaterialProperty() = default;

  virtual ~LinearXMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const override
  {
    auto const one = dealii::make_vectorized_array<double>(1.);
    return one + std::abs(p[0]);
  }

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override
  {
    return 1. + std::abs(p[0]);
  }
};

template <int dim>
class LinearMaterialProperty final : public Coefficient<dim>
{
public:
  LinearMaterialProperty() = default;

  virtual ~LinearMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const override
  {
    auto const one = dealii::make_vectorized_array<double>(1.);
    auto val = one;
    for (unsigned int d = 0; d < dim; ++d)
      val += (one + static_cast<double>(d) * one) * std::abs(p[d]);

    return val;
  }

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override
  {
    double val = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      val += (1. + d) * std::abs(p[d]);

    return val;
  }
};

template <int dim>
class DiscontinuousMaterialProperty final : public Coefficient<dim>
{
public:
  DiscontinuousMaterialProperty() = default;

  virtual ~DiscontinuousMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const override
  {
    auto dim_scale = dealii::make_vectorized_array<double>(0);
    for (unsigned int i = 0;
         i < dealii::VectorizedArray<double>::n_array_elements; ++i)
      for (unsigned int d = 0; d < dim; ++d)
        dim_scale[i] +=
            static_cast<unsigned int>(std::floor(100. * p[d][i])) % 2;

    dealii::VectorizedArray<double> return_value;
    for (unsigned int i = 0;
         i < dealii::VectorizedArray<double>::n_array_elements; ++i)
      return_value[i] = (dim_scale[i] == dim ? 100. : 10.);

    return return_value;
  }

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override

  {
    unsigned int dim_scale = 0;
    for (unsigned int d = 0; d < dim; ++d)
      dim_scale += static_cast<unsigned int>(std::floor(p[d] * 100)) % 2;

    return (dim_scale == dim ? 100. : 10.);
  }
};

template <int dim>
class MaterialPropertyFactory
{
public:
  static std::shared_ptr<Coefficient<dim>>
  create_material_property(std::string const &material_type)
  {
    if (material_type == "constant")
      return std::make_shared<ConstantMaterialProperty<dim>>();
    else if (material_type == "linear_x")
      return std::make_shared<LinearXMaterialProperty<dim>>();
    else if (material_type == "linear")
      return std::make_shared<LinearMaterialProperty<dim>>();
    else if (material_type == "discontinuous")
      return std::make_shared<DiscontinuousMaterialProperty<dim>>();
    else
    {
      mfmg::ASSERT_THROW_NOT_IMPLEMENTED();

      return nullptr;
    }
  }
};

#endif
