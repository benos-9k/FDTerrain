
#ifndef INITIAL3D_FLOAT3_HPP
#define INITIAL3D_FLOAT3_HPP

#include <iostream>
#include <cmath>

#include <xmmintrin.h>

namespace initial3d {

	// sse vector
	class float3 {
	private:
		__m128 m_data;

	public:
		inline float3(__m128 data_) : m_data(data_) { }

		inline float3() : m_data(_mm_setzero_ps()) { }

		inline float3(float v) : m_data(_mm_set1_ps(v)) { }

		inline float3(float x, float y, float z) : m_data(_mm_set_ps(0, z, y, x)) { }

		inline __m128 data() const {
			return m_data;
		}

		inline float x() const {
			float r;
			_mm_store_ss(&r, m_data);
			return r;
		}

		inline float y() const {
			float r;
			_mm_store_ss(&r, _mm_shuffle_ps(m_data, m_data, _MM_SHUFFLE(0, 0, 0, 1)));
			return r;
		}

		inline float z() const {
			float r;
			_mm_store_ss(&r, _mm_shuffle_ps(m_data, m_data, _MM_SHUFFLE(0, 0, 0, 2)));
			return r;
		}

		inline float3 operator-() const {
			return float3(_mm_sub_ps(_mm_setzero_ps(), m_data));
		}

		inline float3 & operator+=(const float3 &rhs) {
			m_data = _mm_add_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator+=(float rhs) {
			m_data = _mm_add_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator+(const float3 &rhs) const {
			return float3(*this) += rhs;
		}

		inline float3 operator+(float rhs) const {
			return float3(*this) += rhs;
		}

		inline friend float3 operator+(float lhs, const float3 &rhs) {
			return float3(rhs) + lhs;
		}

		inline float3 & operator-=(const float3 &rhs) {
			m_data = _mm_sub_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator-=(float rhs) {
			m_data = _mm_sub_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator-(const float3 &rhs) const {
			return float3(*this) -= rhs;
		}

		inline float3 operator-(float rhs) const {
			return float3(*this) -= rhs;
		}

		inline friend float3 operator-(float lhs, const float3 &rhs) {
			return float3(_mm_set1_ps(lhs)) -= rhs;
		}

		inline float3 & operator*=(const float3 &rhs) {
			m_data = _mm_mul_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator*=(float rhs) {
			m_data = _mm_mul_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator*(const float3 &rhs) const {
			return float3(*this) *= rhs;
		}

		inline float3 operator*(float rhs) const {
			return float3(*this) *= rhs;
		}

		inline friend float3 operator*(float lhs, const float3 &rhs) {
			return float3(rhs) * lhs;
		}

		inline float3 & operator/=(const float3 &rhs) {
			m_data = _mm_div_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator/=(float rhs) {
			m_data = _mm_div_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator/(const float3 &rhs) const {
			return float3(*this) /= rhs;
		}

		inline float3 operator/(float rhs) const {
			return float3(*this) /= rhs;
		}

		inline friend float3 operator/(float lhs, const float3 &rhs) {
			return float3(_mm_set1_ps(lhs)) /= rhs;
		}

		inline float3 operator<(const float3 &rhs) const {
			return float3(_mm_cmplt_ps(m_data, rhs.m_data));
		}

		inline float3 operator<=(const float3 &rhs) const {
			return float3(_mm_cmple_ps(m_data, rhs.m_data));
		}

		inline float3 operator>(const float3 &rhs) const {
			return float3(_mm_cmpgt_ps(m_data, rhs.m_data));
		}

		inline float3 operator>=(const float3 &rhs) const {
			return float3(_mm_cmpge_ps(m_data, rhs.m_data));
		}

		inline float3 operator==(const float3 &rhs) const {
			return float3(_mm_cmpeq_ps(m_data, rhs.m_data));
		}

		inline float3 operator!=(const float3 &rhs) const {
			return float3(_mm_cmpneq_ps(m_data, rhs.m_data));
		}

		inline friend std::ostream & operator<<(std::ostream &out, const float3 &v) {
			out << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
			return out;
		}

		inline static float3 abs(const float3 &x) {
			return float3(_mm_andnot_ps(_mm_set1_ps(-0.f), x.m_data));
		}

		inline static float3 max(const float3 &x, const float3 &y) {
			return float3(_mm_max_ps(x.m_data, y.m_data));
		}

		inline static float3 min(const float3 &x, const float3 &y) {
			return float3(_mm_min_ps(x.m_data, y.m_data));
		}

		inline static float3 lerp(const float3 &a, const float3 &b, float t) {
			return a * (1.f - t) + b * t;
		}

		inline static float dot(const float3 &lhs, const float3 &rhs) {
			__m128 r1 = _mm_mul_ps(lhs.m_data, rhs.m_data);
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_add_ps(r3, _mm_add_ps(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r;
		}
		
		inline static float3 cross(const float3 &lhs, const float3 &rhs) {
			static const unsigned shuf_xzy = _MM_SHUFFLE(0, 0, 2, 1);
			static const unsigned shuf_yxz = _MM_SHUFFLE(0, 1, 0, 2);
			__m128 r1 = _mm_mul_ps(_mm_shuffle_ps(lhs.m_data, lhs.m_data, shuf_xzy), _mm_shuffle_ps(rhs.m_data, rhs.m_data, shuf_yxz));
			__m128 r2 = _mm_mul_ps(_mm_shuffle_ps(lhs.m_data, lhs.m_data, shuf_yxz), _mm_shuffle_ps(rhs.m_data, rhs.m_data, shuf_xzy));
			return float3(_mm_sub_ps(r1, r2));
		}
		
		inline static bool all(const float3 &x) {
			__m128 r1 = x.m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_and_ps(r3, _mm_and_ps(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r != 0.f;
		}

		inline static bool any(const float3 &x) {
			__m128 r1 = x.m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_or_ps(r3, _mm_or_ps(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r != 0.f;
		}

		inline float min() const {
			__m128 r1 = m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_min_ss(r3, _mm_min_ss(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r;
		}

		inline float max() const {
			__m128 r1 = m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_max_ss(r3, _mm_max_ss(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r;
		}

		inline bool isnan() const {
			return any(float3(_mm_cmpunord_ps(m_data, m_data)));
		}

		inline float mag() const {
			return sqrt(dot(*this, *this));
		}

		inline float3 unit() const {
			float a = dot(*this, *this);
			__m128 r1 = _mm_rsqrt_ss(_mm_load_ss(&a));
			__m128 r2 = _mm_mul_ps(m_data, _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 0)));
			return float3(r2);
		}

	};

}

#endif
