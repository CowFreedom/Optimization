#pragma once

extern "C"
{
	void calc_stepdirection_f32(int rdim, int xdim, const float* xi_h, const float* residual_h, const float* J_h, float* output_h,bool* lu_used);
	
}