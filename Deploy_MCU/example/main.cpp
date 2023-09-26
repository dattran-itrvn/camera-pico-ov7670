/**
 * Copyright (c) 2022 Brian Starkey <stark3y@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// include C IO 
#include <stdio.h>

// include of pico hardware
#include "hardware/i2c.h"
#include "pico/stdio.h"
#include "pico/stdlib.h"

// include of ov camera
#include "../include/camera/camera.h"
#include "../include/camera/format.h"

// include of tflite framework
#include "../model/tflite_model.h"
#include "../model/ml_model.h"

// Config IO and PIO of PICO
#define CAMERA_PIO      pio0
#define CAMERA_BASE_PIN 10
#define CAMERA_XCLK_PIN 21
#define CAMERA_SDA      0
#define CAMERA_SCL      1

// Model Size
MLModel ml_model(tflite_model, 150 * 1024);

static inline int __i2c_write_blocking(void *i2c_handle, uint8_t addr, const uint8_t *src, size_t len)
{
	return i2c_write_blocking((i2c_inst_t *)i2c_handle, addr, src, len, false);
}

static inline int __i2c_read_blocking(void *i2c_handle, uint8_t addr, uint8_t *dst, size_t len)
{
	return i2c_read_blocking((i2c_inst_t *)i2c_handle, addr, dst, len, false);
}

int8_t* data_input = nullptr;
float scale;
int32_t zero_point;

int main() {
	stdio_init_all();

	// Wait some time for USB serial connection
	sleep_ms(3000);

    printf("-- Pico Tiny-ML-ITRVN detection --\n");

    if (!ml_model.init()) {
        printf("Failed to initialize ML model!\n");
        while (1) { tight_loop_contents(); }
    }
    else {
        printf("|    ML model initialize OK    |\n");
		data_input = (int8_t*)ml_model.input_data();
		printf("Size input: %d, ", sizeof(data_input));
		scale = ml_model.input_scale(); 
		printf("Scale: %f, ",scale);
		zero_point = ml_model.input_zero_point();
		printf("Zero Point: %d\r\n", zero_point);

    }

	i2c_init(i2c0, 100000);
	gpio_set_function(CAMERA_SDA, GPIO_FUNC_I2C);
	gpio_set_function(CAMERA_SCL, GPIO_FUNC_I2C);
	gpio_pull_up(CAMERA_SDA);
	gpio_pull_up(CAMERA_SCL);

	struct camera camera;
	struct camera_platform_config platform = {
		.i2c_write_blocking = __i2c_write_blocking,
		.i2c_read_blocking = __i2c_read_blocking,
		.i2c_handle = i2c0,

		.pio = CAMERA_PIO,
		.xclk_pin = CAMERA_XCLK_PIN,
		.xclk_divider = 9,
		.base_pin = CAMERA_BASE_PIN,
		.base_dma_channel = -1,
	};

	int ret = camera_init(&camera, &platform);
	if (ret) {
		printf("camera_init failed: %d\n", ret);
		return 1;
	}
	else{
		printf("|       camera_init OK        |\r\n");

	}
	printf("=============================\r\n");

	const uint16_t width = CAMERA_WIDTH_DIV8;
	const uint16_t height = CAMERA_HEIGHT_DIV8;

	struct camera_buffer *buf = camera_buffer_alloc(FORMAT_YUV422, width, height);
	assert(buf);
	int frame_id = 0;

	while (1) {
		// printf("[%03dx%03d] %04d$", width, height, frame_id);
		ret = camera_capture_blocking(&camera, buf, true);
		if (ret != 0) {
			printf("Capture error: %d\n", ret);
		} else {
			int y, x;
			for (y = 0; y < height; y++) {
				char row[width];
				for (x = 0; x < width; x++) {
					data_input[buf->strides[0] * y + x] = buf->data[0][buf->strides[0] * y + x] / 
														  scale + zero_point;
					// char snum[4];
    				// int n = sprintf(snum, "%d", buf->data[0][buf->strides[0] * y + x]);
					// printf(" %s", snum);
				}
			}
			printf("\n");
			uint64_t start = time_us_64();
			float prediction = ml_model.predict();
			uint32_t time_taken = time_us_64() - start;;
			printf("Predict: %f == Time: %d us\n", prediction, time_taken);
			memset(data_input, 0, 60 * 60);

			frame_id++;
			if (frame_id >= 1000)
				frame_id = 0;
		}

		sleep_ms(2000);
	}
}
