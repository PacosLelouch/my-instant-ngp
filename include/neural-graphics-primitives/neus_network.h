/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/network_with_input_encoding.h>

#define NEUS_DEBUG_BACKWARD 1

NGP_NAMESPACE_BEGIN

NGP_NEUS_NAMESPACE_BEGIN

template <typename T>
__global__ void extract_sdf_value(
	const uint32_t n_elements,
	const uint32_t sdf_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ sdf_value,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd[i * rgbd_stride] = sdf_value[i * sdf_stride];
}

template <typename T, typename TIn = T>
__global__ void extract_dSDF_dPos(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const uint32_t output_stride,
	const TIn* __restrict__ dSDF_dPos,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgbd[elem_idx*rgb_stride + dim_idx] = dSDF_dPos[elem_idx*output_stride + dim_idx];
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const uint32_t output_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*output_stride + dim_idx];
}

// TODO: TODO: is rgbd [4]?
template <typename T>
__global__ void add_density_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T, typename TInput = T>
__global__ void fill_positions(
	const uint32_t n_elements,
	const uint32_t pos_stride,
	const uint32_t dst_stride,
	const uint32_t n_pos_dim,
	const TInput* __restrict__ pos,
	T* __restrict__ dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	dst[elem_idx*dst_stride + dim_idx] = pos[elem_idx*pos_stride + dim_idx];
}

template <typename T>
__global__ void set_constant_value(
	const uint32_t n_elements,
	const uint32_t output_stride,
	const T value,
	T* __restrict__ output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	output[i * output_stride] = value;
}

//// Start: New SingleVarianceNetwork
//template <typename T>
//class SingleVarianceNetwork : public tcnn::Network<float, T> {
//public:
//	using json = nlohmann::json;
//
//	SingleVarianceNetwork(float init_val) {
//		// TODO
//	}
//
//	virtual ~SingleVarianceNetwork() {}
//
//	virtual void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
//		uint32_t batch_size = input.n();
//		// TODO
//	}
//
//	virtual std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
//		// TODO
//	}
//
//	virtual void backward_impl(
//		cudaStream_t stream,
//		const tcnn::Context& ctx,
//		const tcnn::GPUMatrixDynamic<float>& input,
//		const tcnn::GPUMatrixDynamic<T>& output,
//		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
//		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
//		bool use_inference_params = false,
//		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
//	) override {
//		// TODO
//	}
//
//	virtual void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
//		// TODO
//	}
//
//	virtual void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
//		// TODO
//	}
//
//	// Begin: attribute override.
//
//	size_t n_params() const override {
//		return m_pos_encoding->n_params() + m_sdf_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
//	}
//
//	uint32_t padded_output_width() const override {
//		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
//	}
//
//	uint32_t input_width() const override {
//		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
//	}
//
//	uint32_t output_width() const override {
//		return 1;
//	}
//
//	uint32_t required_input_alignment() const override {
//		return 1; // No alignment required due to encoding
//	}
//
//	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
//		auto layers = m_sdf_network->layer_sizes();
//		auto rgb_layers = m_rgb_network->layer_sizes();
//		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
//		return layers;
//	}
//
//	uint32_t width(uint32_t layer) const override {
//		if (layer == 0) {
//			return m_pos_encoding->padded_output_width();
//		} else if (layer < m_sdf_network->num_forward_activations() + 1) {
//			return m_sdf_network->width(layer - 1);
//		} else if (layer == m_sdf_network->num_forward_activations() + 1) {
//			return m_rgb_network_input_width;
//		} else {
//			return m_rgb_network->width(layer - 2 - m_sdf_network->num_forward_activations());
//		}
//	}
//
//	uint32_t num_forward_activations() const override {
//		return m_sdf_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
//	}
//
//	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
//		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
//		if (layer == 0) {
//			return {forward.sdf_network_input.data(), m_pos_encoding->preferred_output_layout()};
//		} else if (layer < m_sdf_network->num_forward_activations() + 1) {
//			return m_sdf_network->forward_activations(*forward.sdf_network_ctx, layer - 1);
//		} else if (layer == m_sdf_network->num_forward_activations() + 1) {
//			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
//		} else {
//			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_sdf_network->num_forward_activations());
//		}
//	}
//
//	tcnn::json hyperparams() const override {
//		return {
//			{"otype", "SingleVarianceNetwork"},
//			{"init_val", m_init_val},
//		};
//	}
//
//	// End: attribute override.
//
//
//protected:
//	//uint32_t m_n_input_dims;
//	float m_init_val = 0.3;
//
//	// // Storage of forward pass data
//	struct ForwardContext : public tcnn::Context {
//		//tcnn::GPUMatrixDynamic<T> sdf_network_input;
//		//tcnn::GPUMatrixDynamic<T> sdf_network_output;
//
//		//tcnn::GPUMatrixDynamic<T> rgb_network_input;
//		//tcnn::GPUMatrix<T> rgb_network_output;
//
//		//std::unique_ptr<Context> pos_encoding_ctx;
//		//std::unique_ptr<Context> dir_encoding_ctx;
//
//		//std::unique_ptr<Context> sdf_network_ctx;
//		//std::unique_ptr<Context> rgb_network_ctx;
//	};
//};
//// End: New SingleVarianceNetwork

// Different network structure.
template <typename T>
class NeusNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NeusNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& sdf_network, const json& rgb_network)
		: m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, sdf_network.contains("otype") && (tcnn::equals_case_insensitive(sdf_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(sdf_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_sdf_network_config = sdf_network; // TODO: sdf_network scale (float).
		local_sdf_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		//if (!sdf_network.contains("n_output_dims")) {
		//	local_sdf_network_config["n_output_dims"] = m_pos_encoding->padded_output_width();//16;
		//}
		m_sdf_network.reset(tcnn::create_network<T>(local_sdf_network_config));

		m_sdf_feature_output_width = tcnn::next_multiple(m_sdf_network->padded_output_width(), rgb_alignment) - rgb_alignment;//m_sdf_network->padded_output_width() / 2;

		//// Init sdf_network_feature config.
		//json local_sdf_network_feature_config = sdf_network_feature;
		//local_sdf_network_feature_config["n_input_dims"] = local_sdf_network_config["n_output_dims"];
		//if (!sdf_network_feature.contains("n_output_dims")) {
		//	local_sdf_network_feature_config["n_output_dims"] = m_pos_encoding->padded_output_width();//16;
		//}
		//m_sdf_feature_network.reset(tcnn::create_network<T>(local_sdf_network_feature_config));

		//// Init sdf_network_sdf config.
		//json local_sdf_network_sdf_config = sdf_network_sdf;
		//local_sdf_network_sdf_config["n_input_dims"] = local_sdf_network_config["n_output_dims"];
		//if (!sdf_network_sdf.contains("n_output_dims")) {
		//	local_sdf_network_sdf_config["n_output_dims"] = 16;
		//}
		//m_sdf_sdf_network.reset(tcnn::create_network<T>(local_sdf_network_sdf_config));

		// sdf gradient on pos keeps the same width of pos. Pos not encoded.
		m_rgb_network_input_width = tcnn::next_multiple(m_n_pos_dims + m_n_pos_dims + m_dir_encoding->padded_output_width() + m_sdf_feature_output_width, rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
	}

	virtual ~NeusNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		forward_impl(stream, input, &output, use_inference_params, true);
		return; // Since we need gradient in inference. Should it be replaced by forward_impl?

		//uint32_t batch_size = input.n();
		//tcnn::GPUMatrixDynamic<T> sdf_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		//tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		//// Same as sdf_feature_network_input and sdf_sdf_network_input.
		//tcnn::GPUMatrixDynamic<T> sdf_network_output{ m_sdf_network->padded_output_width(), batch_size, stream, m_sdf_network->preferred_output_layout() }; 

		//tcnn::GPUMatrixDynamic<T> sdf_feature_network_output = rgb_network_input.slice_rows(0, m_sdf_feature_network->padded_output_width());
		//tcnn::GPUMatrixDynamic<T> sdf_sdf_network_output{ m_sdf_sdf_network->padded_output_width(), batch_size, stream, m_sdf_sdf_network->preferred_output_layout() }; // Extra memory.
		//tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		//m_pos_encoding->inference_mixed_precision(
		//	stream,
		//	input.slice_rows(0, m_pos_encoding->input_width()),
		//	sdf_network_input,
		//	use_inference_params
		//);

		//m_sdf_network->inference_mixed_precision(stream, sdf_network_input, sdf_network_output, use_inference_params);

		//// SDF outputs.
		//m_sdf_feature_network->inference_mixed_precision(stream, sdf_network_output, sdf_feature_network_output, use_inference_params);
		//m_sdf_sdf_network->inference_mixed_precision(stream, sdf_network_output, sdf_sdf_network_output, use_inference_params);

		//// TODO: TODO: Get SDF gradient from positions.

		//auto dir_out = rgb_network_input.slice_rows(m_sdf_network->padded_output_width(), m_dir_encoding->padded_output_width());
		//m_dir_encoding->inference_mixed_precision(
		//	stream,
		//	input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
		//	dir_out,
		//	use_inference_params
		//);

		//m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		//tcnn::linear_kernel(extract_sdf_value<T>, 0, stream,
		//	batch_size,
		//	sdf_network_output.layout() == tcnn::AoS ? sdf_network_output.stride() : 1,
		//	output.layout() == tcnn::AoS ? padded_output_width() : 1,
		//	sdf_network_output.data(),
		//	output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		//);
	}

	uint32_t padded_density_output_width() const {
		return m_sdf_network->padded_output_width();
	}

	// TODO: Change to NeUS.
	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		/// Begin: Get sdf.
		forward->sdf_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		// Position encoding.
		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->sdf_network_input,
			use_inference_params,
			true//prepare_input_gradients
		);

		forward->sdf_network_output = tcnn::GPUMatrixDynamic<T>{ m_sdf_network->padded_output_width(), batch_size, stream, forward->rgb_network_input.layout() }; 
		//forward->sdf_network_output = forward->rgb_network_input.slice_rows(0, m_sdf_network->padded_output_width());
		forward->sdf_network_ctx = m_sdf_network->forward(
			stream,
			forward->sdf_network_input,
			&forward->sdf_network_output,
			use_inference_params,
			true//prepare_input_gradients
			);
		/// End: Get sdf.
		
		/// Begin: Get sdf gradient.
		tcnn::GPUMatrixDynamic<T> dSDF_dSDF{ m_sdf_network->padded_output_width(), batch_size, stream, forward->rgb_network_input.layout() };
		dSDF_dSDF.memset(0);
		tcnn::linear_kernel(set_constant_value<T>, 0, stream,
			batch_size, dSDF_dSDF.m(), 1.0f, dSDF_dSDF.data() + m_sdf_feature_output_width);
		////tcnn::linear_kernel(set_constant_value<T>, 0, stream,
		////	batch_size, dSDF_dSDF.m(), 1.0f, dSDF_dSDF.data());
		tcnn::GPUMatrixDynamic<T> dSDF_dPosEncoding{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };

		//tcnn::GPUMatrixDynamic<float> dSDF_dInput{ input.m(), batch_size, stream, input.layout() };
		//tcnn::GPUMatrixDynamic<float> dSDF_dPos = dSDF_dInput.slice_rows(0, m_pos_encoding->input_width());

		tcnn::GPUMatrixDynamic<float> dSDF_dPos{ m_pos_encoding->input_width(), batch_size, stream, input.layout() }; // = gradient.

		m_sdf_network->backward(stream, *forward->sdf_network_ctx, forward->sdf_network_input, forward->sdf_network_output, dSDF_dSDF, &dSDF_dPosEncoding, use_inference_params, EGradientMode::Ignore);
		//CUDA_CHECK_THROW(cudaDeviceSynchronize());
		
		m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, input.slice_rows(0, m_pos_encoding->input_width()), forward->sdf_network_input, dSDF_dPosEncoding, &dSDF_dPos, use_inference_params, EGradientMode::Ignore);
		//CUDA_CHECK_THROW(cudaDeviceSynchronize());

		/// End: Get sdf gradient.

		// rgb_input: feature + pos + gradient + dir_encoding.
		tcnn::linear_kernel(fill_positions<T, float>, 0, stream,
			batch_size, m_pos_encoding->input_width(), m_rgb_network_input_width, m_pos_encoding->input_width(),
			input.slice_rows(0, m_pos_encoding->input_width()).data(), forward->rgb_network_input.slice_rows(m_sdf_feature_output_width, m_pos_encoding->input_width()).data());

		tcnn::linear_kernel(fill_positions<T, float>, 0, stream,
			batch_size, m_pos_encoding->input_width(), m_rgb_network_input_width, m_pos_encoding->input_width(),
			dSDF_dPos.data(), forward->rgb_network_input.slice_rows(m_sdf_feature_output_width + m_pos_encoding->input_width(), m_pos_encoding->input_width()).data());

		auto dir_out = forward->rgb_network_input.slice_rows(m_sdf_feature_output_width + m_pos_encoding->input_width() + m_pos_encoding->input_width(), m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {

#if NEUS_DEBUG_BACKWARD
			if (output && output->layout() == tcnn::AoS) {
				// Begin: Debug GPU
				std::vector<T> output_CPU_debug(output->m());
				CUDA_CHECK_THROW(cudaMemcpyAsync(output_CPU_debug.data(), output->data(), sizeof(T) * output->m(), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
				std::vector<float> output_CPU_debug_float(output->m());
				for (size_t i = 0; i < output_CPU_debug.size(); ++i) {
					output_CPU_debug_float[i] = float(output_CPU_debug[i]);
				}
				std::cout << (float)output_CPU_debug_float[0] << std::endl;
				// End: Debug GPU
			}
#endif // NEUS_DEBUG_BACKWARD

			// tcnn::GPUMatrixDynamic<T>(data, m, n, layout)
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);


		if (output) {
#if NEUS_DEBUG_BACKWARD
			// Begin: Debug GPU
			if (output && output->layout() == tcnn::AoS) {
				std::vector<T> rgb_network_output_CPU_debug(forward->rgb_network_output.m());
				CUDA_CHECK_THROW(cudaMemcpyAsync(rgb_network_output_CPU_debug.data(), forward->rgb_network_output.data(), sizeof(T) * forward->rgb_network_output.m(), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
				std::vector<float> rgb_network_output_CPU_debug_float(forward->rgb_network_output.m());
				for (size_t i = 0; i < rgb_network_output_CPU_debug.size(); ++i) {
					rgb_network_output_CPU_debug_float[i] = float(rgb_network_output_CPU_debug[i]);
				}
				std::cout << (float)rgb_network_output_CPU_debug_float[0] << std::endl;
			}
			// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD
			tcnn::linear_kernel(extract_sdf_value<T>, 0, stream,
				batch_size,
				m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->sdf_network_output.stride() : 1,
				output->layout() == tcnn::AoS ? padded_output_width() : 1,
				forward->sdf_network_output.data() + m_sdf_feature_output_width,
				output->data() + m_pos_encoding->input_width() * (output->layout() == tcnn::AoS ? 1 : batch_size)
			);
			tcnn::linear_kernel(extract_dSDF_dPos<T, float>, 0, stream,
				batch_size*3,
				m_pos_encoding->input_width(),
				output->layout() == tcnn::AoS ? padded_output_width() : 1,
				dSDF_dPos.data(),
				output->data() + (1 + m_pos_encoding->input_width()) * (output->layout() == tcnn::AoS ? 1 : batch_size)
			);

#if NEUS_DEBUG_BACKWARD
			// Begin: Debug GPU
			if (output && output->layout() == tcnn::AoS) {
				std::vector<T> output_CPU_debug(output->m());
				CUDA_CHECK_THROW(cudaMemcpyAsync(output_CPU_debug.data(), output->data(), sizeof(T) * output->m(), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
				std::vector<float> output_CPU_debug_float(output->m());
				for (size_t i = 0; i < output_CPU_debug.size(); ++i) {
					output_CPU_debug_float[i] = float(output_CPU_debug[i]);
				}
				std::cout << (float)output_CPU_debug_float[0] << std::endl;
			}
			// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD
		}

		return forward;
	}

	// TODO: Change to NeUS.
	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

#if NEUS_DEBUG_BACKWARD
		// Begin: Debug GPU
		std::vector<T> dL_drgb_CPU_debug(m_rgb_network->padded_output_width());
		CUDA_CHECK_THROW(cudaMemcpyAsync(dL_drgb_CPU_debug.data(), dL_drgb.data(), sizeof(T) * m_rgb_network->padded_output_width(), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		std::vector<float> dL_drgb_CPU_debug_float(m_rgb_network->padded_output_width());
		for (size_t i = 0; i < dL_drgb_CPU_debug.size(); ++i) {
			dL_drgb_CPU_debug_float[i] = float(dL_drgb_CPU_debug[i]);
		}
		std::cout << (float)dL_drgb_CPU_debug_float[0] << std::endl;
		// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

#if NEUS_DEBUG_BACKWARD
		// Begin: Debug GPU
		std::vector<T> dL_drgb_network_input_CPU_debug(m_rgb_network_input_width);
		CUDA_CHECK_THROW(cudaMemcpyAsync(dL_drgb_network_input_CPU_debug.data(), dL_drgb_network_input.data(), sizeof(T) * m_rgb_network_input_width, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		std::vector<float> dL_drgb_network_input_CPU_debug_float(m_rgb_network_input_width);
		for (size_t i = 0; i < dL_drgb_network_input_CPU_debug.size(); ++i) {
			dL_drgb_network_input_CPU_debug_float[i] = float(dL_drgb_network_input_CPU_debug[i]);
		}
		std::cout << (float)dL_drgb_network_input_CPU_debug_float[0] << std::endl;
		// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_sdf_network->padded_output_width(), m_dir_encoding->padded_output_width());
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_sdf_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);

#if NEUS_DEBUG_BACKWARD
			// Begin: Debug GPU
			std::vector<T> dL_ddir_encoding_input_CPU_debug(m_dir_encoding->input_width());
			CUDA_CHECK_THROW(cudaMemcpyAsync(dL_ddir_encoding_input_CPU_debug.data(), dL_ddir_encoding_input.data(), sizeof(T) * m_dir_encoding->input_width(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			std::vector<float> dL_ddir_encoding_input_CPU_debug_float(m_dir_encoding->input_width());
			for (size_t i = 0; i < dL_ddir_encoding_input_CPU_debug.size(); ++i) {
				dL_ddir_encoding_input_CPU_debug_float[i] = float(dL_ddir_encoding_input_CPU_debug[i]);
			}
			std::cout << (float)dL_ddir_encoding_input_CPU_debug_float[0] << std::endl;
			// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD
		}

		tcnn::GPUMatrixDynamic<T> dL_dsdf_network_output = dL_drgb_network_input.slice_rows(0, m_sdf_network->padded_output_width());
		tcnn::linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_dsdf_network_output.layout() == tcnn::RM ? 1 : dL_dsdf_network_output.stride(),
			dL_dsdf_network_output.data()
		);

#if NEUS_DEBUG_BACKWARD
		// Begin: Debug GPU
		std::vector<T> dL_dsdf_network_output_CPU_debug(m_sdf_network->padded_output_width());
		CUDA_CHECK_THROW(cudaMemcpyAsync(dL_dsdf_network_output_CPU_debug.data(), dL_dsdf_network_output.data(), sizeof(T) * m_sdf_network->padded_output_width(), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		std::vector<float> dL_dsdf_network_output_CPU_debug_float(m_sdf_network->padded_output_width());
		for (size_t i = 0; i < dL_dsdf_network_output_CPU_debug.size(); ++i) {
			dL_dsdf_network_output_CPU_debug_float[i] = float(dL_dsdf_network_output_CPU_debug[i]);
		}
		std::cout << (float)dL_dsdf_network_output_CPU_debug_float[0] << std::endl;
		// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD

		tcnn::GPUMatrixDynamic<T> dL_dsdf_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_dsdf_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_sdf_network->backward(stream, *forward.sdf_network_ctx, forward.sdf_network_input, forward.sdf_network_output, dL_dsdf_network_output, dL_dsdf_network_input.data() ? &dL_dsdf_network_input : nullptr, use_inference_params, param_gradients_mode);

#if NEUS_DEBUG_BACKWARD
		// Begin: Debug GPU
		if (dL_dsdf_network_input.m() > 0) {
			std::vector<T> dL_dsdf_network_input_CPU_debug(m_pos_encoding->padded_output_width());
			CUDA_CHECK_THROW(cudaMemcpyAsync(dL_dsdf_network_input_CPU_debug.data(), dL_dsdf_network_input.data(), sizeof(T) * m_pos_encoding->padded_output_width(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			std::vector<float> dL_dsdf_network_input_CPU_debug_float(m_pos_encoding->padded_output_width());
			for (size_t i = 0; i < dL_dsdf_network_input_CPU_debug.size(); ++i) {
				dL_dsdf_network_input_CPU_debug_float[i] = float(dL_dsdf_network_input_CPU_debug[i]);
			}
			std::cout << (float)dL_dsdf_network_input_CPU_debug_float[0] << std::endl;
		}
		// End: Debug GPU
#endif // NEUS_DEBUG_BACKWARD

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_dsdf_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.sdf_network_input,
				dL_dsdf_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);

#if NEUS_DEBUG_BACKWARD
			if (dL_dinput) {
				// Begin: Debug GPU
				std::vector<float> dL_dpos_encoding_input_CPU_debug(m_pos_encoding->input_width());
				CUDA_CHECK_THROW(cudaMemcpyAsync(dL_dpos_encoding_input_CPU_debug.data(), dL_dpos_encoding_input.data(), sizeof(float) * m_pos_encoding->input_width(), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
				std::cout << (float)dL_dpos_encoding_input_CPU_debug[0] << std::endl;
				// End: Debug GPU
			}
#endif // NEUS_DEBUG_BACKWARD
		}
	}

	// TODO: TODO: density -> sdf
	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NeusNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		tcnn::GPUMatrixDynamic<T> sdf_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			sdf_network_input,
			use_inference_params
		);

		m_sdf_network->inference_mixed_precision(stream, sdf_network_input, output, use_inference_params);
	}

	std::unique_ptr<tcnn::Context> density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NeusNetwork::density_forward input must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->sdf_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->sdf_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->sdf_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_sdf_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->sdf_network_ctx = m_sdf_network->forward(stream, forward->sdf_network_input, output ? &forward->sdf_network_output : nullptr, use_inference_params, prepare_input_gradients);

		return forward;
	}

	// TODO: TODO: density -> sdf
	void density_backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) {
		if (input.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
			throw std::runtime_error("NeusNetwork::density_backward input must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrixDynamic<T> dL_dsdf_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_dsdf_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_sdf_network->backward(stream, *forward.sdf_network_ctx, forward.sdf_network_input, output, dL_doutput, dL_dsdf_network_input.data() ? &dL_dsdf_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_dsdf_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.sdf_network_input,
				dL_dsdf_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		m_sdf_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_sdf_network->n_params();

		m_rgb_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_dir_encoding->n_params();
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_sdf_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_sdf_network->n_params();

		m_rgb_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_sdf_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)8);//4
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t output_width() const override {
		return 7;//4
	}

	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_sdf_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->padded_output_width();
		} else if (layer < m_sdf_network->num_forward_activations() + 1) {
			return m_sdf_network->width(layer - 1);
		} else if (layer == m_sdf_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_sdf_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_sdf_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.sdf_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_sdf_network->num_forward_activations() + 1) {
			return m_sdf_network->forward_activations(*forward.sdf_network_ctx, layer - 1);
		} else if (layer == m_sdf_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_sdf_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	const std::unique_ptr<tcnn::Network<T>>& sdf_network() const {
		return m_sdf_network;
	}

	tcnn::json hyperparams() const override {
		json sdf_network_hyperparams = m_sdf_network->hyperparams();
		sdf_network_hyperparams["n_output_dims"] = m_sdf_network->padded_output_width();
		return {
			{"otype", "NeusNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"sdf_network", sdf_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
		};
	}

private:
	// TODO: TODO: density -> sdf hidden
	std::unique_ptr<tcnn::Network<T>> m_sdf_network;
	std::unique_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	uint32_t m_sdf_feature_output_width;
	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	// // Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		// TODO: TODO: density -> sdf hidden
		tcnn::GPUMatrixDynamic<T> sdf_network_input;
		tcnn::GPUMatrixDynamic<T> sdf_network_output;

		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrixDynamic<T> rgb_network_output;
		//tcnn::GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		// TODO: TODO: density -> sdf hidden
		std::unique_ptr<Context> sdf_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
	};
};

NGP_NEUS_NAMESPACE_END

NGP_NAMESPACE_END
