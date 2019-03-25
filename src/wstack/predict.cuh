#ifndef PREDICTCU_H
#define PREDICTCU_H


template <typename FloatType>
struct predict_locs{
    FloatType *u;
    FloatType *v;
    FloatType *w;
    std::complex<FloatType> *vals;
    int number_of_predicts;
};

template <typename FloatType>
struct vis_contribution{

    int *locs_u, *locs_v;
    FloatType *gcf_u;
    FloatType *gcf_v;
};

template <typename FloatType>
struct w_plane_locs{
    std::size_t wpi; // W-Plane indicator
    std::size_t num_contribs;
    std::size_t *contrib_index; // As we partially accumulate visibilities, this is the array address of our vis value.
    struct vis_contribution <FloatType> *visc;
};



__host__ std::complex<double> wstack_predict_cu(double theta,
						double lam,
						const std::vector<double>& points,
						std::vector<double> u,
						std::vector<double> v,
						std::vector<double> w,
						double du,
						double dw,
						int aa_support_uv,
						int aa_support_w,
						double x0,
						struct sep_kernel_data *grid_conv_uv,
						struct sep_kernel_data *grid_conv_w,
						struct sep_kernel_data *grid_corr_lm,
						struct sep_kernel_data *grid_corr_n);


#endif