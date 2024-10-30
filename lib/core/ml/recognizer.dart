import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_absensi_app_2/core/ml/recognition_embedding.dart';
import 'package:flutter_absensi_app_2/data/datasources/auth_local_datasource.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;

  static const int WIDTH = 112;
  static const int HEIGHT = 112;

  String get modelName => 'assets/mobile_face_net.tflite';

  // Constructor untuk Recognizer
  Recognizer({int? numThreads}) {
    // Inisialisasi InterpreterOptions
    try {
      _interpreterOptions =
          InterpreterOptions(); // Pastikan diinisialisasi di sini
      print("InterpreterOptions initialized");

      if (numThreads != null) {
        print("Setting numThreads to: $numThreads");
        _interpreterOptions.threads =
            numThreads; // Mengatur jumlah threads jika disediakan
      } else {
        print("numThreads is null, using default.");
      }
    } catch (e) {
      print("Error initializing InterpreterOptions: ${e.toString()}");
    }

    // Lanjutkan inisialisasi model
    _initializeModel();
  }

  // Inisialisasi model secara async
  Future<void> _initializeModel() async {
    await loadModel();
  }

  // Load model dengan InterpreterOptions
  Future<void> loadModel() async {
    try {
      print("Loading model: $modelName with options: $_interpreterOptions");

      // Inisialisasi Interpreter dengan model dan InterpreterOptions
      interpreter = await Interpreter.fromAsset(
        modelName,
        options: _interpreterOptions,
      );

      print(
          'Interpreter successfully created with options: $_interpreterOptions');
    } catch (e) {
      print('Error creating interpreter: ${e.toString()}');
    }
  }

  // List<dynamic> imageToArray(img.Image inputImage) {
  //   img.Image resizedImage =
  //       img.copyResize(inputImage, width: WIDTH, height: HEIGHT);
  //   List<double> flattenedList = [];
  //     for (int y = 0; y < resizedImage.height; y++) {
  //       for (int x = 0; x < resizedImage.width; x++) {
  //         int pixel = resizedImage.getPixel(x, y) as int;  // dapatkan nilai pixel dalam bentuk int

  //         // Ekstrak komponen warna R, G, dan B dari pixel
  //         int r = (pixel >> 16) & 0xFF; // Shift untuk mendapatkan nilai Red
  //         int g = (pixel >> 8) & 0xFF;  // Shift untuk mendapatkan nilai Green
  //         int b = pixel & 0xFF;         // Ambil nilai Blue langsung

  //         // Tambahkan komponen R, G, dan B ke list
  //         flattenedList.add(r.toDouble());
  //         flattenedList.add(g.toDouble());
  //         flattenedList.add(b.toDouble());
  //       }
  //     }

  //   // Konversi list menjadi Float32List dan lakukan reshape
  //   Float32List float32Array = Float32List.fromList(flattenedList);
  //   int channels = 3;
  //   int height = HEIGHT;
  //   int width = WIDTH;
  //   Float32List reshapedArray = Float32List(1 * height * width * channels);

  //   // Lakukan reshaping array
  //   for (int c = 0; c < channels; c++) {
  //     for (int h = 0; h < height; h++) {
  //       for (int w = 0; w < width; w++) {
  //         int index = c * height * width + h * width + w;
  //         reshapedArray[index] =
  //             (float32Array[c * height * width + h * width + w] - 127.5) / 127.5;
  //       }
  //     }
  //   }

  //   return reshapedArray.reshape([1, 112, 112, 3]);
  // }

  List<dynamic> imageToArray(img.Image inputImage) {
    img.Image resizedImage =
        img.copyResize(inputImage, width: WIDTH, height: HEIGHT);
    List<double> flattenedList = resizedImage.data!
        .expand((channel) => [channel.r, channel.g, channel.b])
        .map((value) => value.toDouble())
        .toList();
    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    int height = HEIGHT;
    int width = WIDTH;
    Float32List reshapedArray = Float32List(1 * height * width * channels);
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] =
              (float32Array[c * height * width + h * width + w] - 127.5) /
                  127.5;
        }
      }
    }
    return reshapedArray.reshape([1, 112, 112, 3]);
  }

  RecognitionEmbedding recognize(img.Image image, Rect location) {
    //TODO crop face from image resize it and convert it to float array
    var input = imageToArray(image);
    print(input.shape.toString());

    //TODO output array
    List output = List.filled(1 * 192, 0).reshape([1, 192]);

    //TODO performs inference
    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    // final run = DateTime.now().millisecondsSinceEpoch - runs;
    // print('Time to run inference: $run ms$output');

    //TODO convert dynamic list to double list
    List<double> outputArray = output.first.cast<double>();

    return RecognitionEmbedding(location, outputArray);
  }

  PairEmbedding findNearest(List<double> emb, List<double> authFaceEmbedding) {
    PairEmbedding pair = PairEmbedding(-5);

    double distance = 0;
    for (int i = 0; i < emb.length; i++) {
      double diff = emb[i] - authFaceEmbedding[i];
      distance += diff * diff;
    }
    distance = sqrt(distance);
    if (pair.distance == -5 || distance < pair.distance) {
      pair.distance = distance;
    }
    //}
    return pair;
  }

  Future<bool> isValidFace(List<double> emb) async {
    final authData = await AuthLocalDatasource().getAuthData();
    final faceEmbedding = authData!.user!.faceEmbedding;
    PairEmbedding pair = findNearest(
        emb,
        faceEmbedding!
            .split(',')
            .map((e) => double.parse(e))
            .toList()
            .cast<double>());
    print("distance= ${pair.distance}");
    if (pair.distance < 1.0) {
      return true;
    }
    return false;
  }
}

class PairEmbedding {
  double distance;
  PairEmbedding(this.distance);
}
