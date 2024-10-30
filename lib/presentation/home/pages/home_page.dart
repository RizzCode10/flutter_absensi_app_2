// import 'package:detect_fake_location/detect_fake_location.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_absensi_app_2/core/helper/radius_calculate.dart';
import 'package:flutter_absensi_app_2/data/datasources/auth_local_datasource.dart';
import 'package:flutter_absensi_app_2/data/models/response/auth_response_model.dart';
import 'package:flutter_absensi_app_2/presentation/home/bloc/get_company/get_company_bloc.dart';
import 'package:flutter_absensi_app_2/presentation/home/bloc/is_checkedin/is_checkedin_bloc.dart';
import 'package:flutter_absensi_app_2/presentation/home/pages/attendance_checkin_page.dart';
import 'package:flutter_absensi_app_2/presentation/home/pages/attendance_checkout_page.dart';
import 'package:flutter_absensi_app_2/presentation/home/pages/permission_page.dart';
import 'package:flutter_absensi_app_2/presentation/home/pages/register_face_attendance_page.dart';
import 'package:flutter_absensi_app_2/presentation/home/pages/setting_page.dart';
import 'package:flutter_bloc/flutter_bloc.dart';


import '../../../core/core.dart';
import '../widgets/menu_button.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? faceEmbedding;

  @override
  void initState() {
    _initializeFaceEmbedding();
    context.read<IsCheckedinBloc>().add(const IsCheckedinEvent.isCheckedIn());
    context.read<GetCompanyBloc>().add(const GetCompanyEvent.getCompany());
    super.initState();
  }

  double? latitude;
  double? longitude;

  Future<void> _initializeFaceEmbedding() async {
    try {
      final authData = await AuthLocalDatasource().getAuthData();
      setState(() {
        faceEmbedding = authData?.user?.faceEmbedding;
      });
    } catch (e) {
      // Tangani error di sini jika ada masalah dalam mendapatkan authData
      print('Error fetching auth data: $e');
      setState(() {
        faceEmbedding = null; // Atur faceEmbedding ke null jika ada kesalahan
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.all(16.0),
          decoration: BoxDecoration(
            image: DecorationImage(
              image: Assets.images.bgHome.provider(),
              alignment: Alignment.topCenter,
            ),
          ),
          child: ListView(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            children: [
              Row(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(50.0),
                    child: Image.network(
                      'https://i.pinimg.com/originals/1b/14/53/1b14536a5f7e70664550df4ccaa5b231.jpg',
                      width: 48.0,
                      height: 48.0,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SpaceWidth(12.0),
                  Expanded(
                    child: FutureBuilder(
                      future: AuthLocalDatasource().getAuthData(),
                      builder: (context, snapshot) {
                        if (snapshot.connectionState ==
                            ConnectionState.waiting) {
                          return const Text('Loading...');
                        }else {
                          final user = snapshot.data?.user;
                          return Text(
                            'Hello, ${user?.name}' ?? 'Hello, Chopper Sensei',
                            style: const TextStyle(
                              fontSize: 18.0,
                              color: AppColors.white,
                            ),
                            maxLines: 2,
                          );
                        }
                      },
                      // child: Text(
                      //   'Hello, Chopper Sensei',
                      //   style: TextStyle(
                      //     fontSize: 18.0,
                      //     color: AppColors.white,
                      //   ),
                      //   maxLines: 2,
                      // ),
                    ),
                  ),
                  IconButton(
                    onPressed: () {},
                    icon: Assets.icons.notificationRounded.svg(),
                  ),
                ],
              ),
              const SpaceHeight(24.0),
              Container(
                padding: const EdgeInsets.all(24.0),
                decoration: BoxDecoration(
                  color: AppColors.white,
                  borderRadius: BorderRadius.circular(20.0),
                ),
                child: Column(
                  children: [
                    Text(
                      DateTime.now().toFormattedTime(),
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 32.0,
                        color: AppColors.primary,
                      ),
                    ),
                    Text(
                      DateTime.now().toFormattedDate(),
                      style: const TextStyle(
                        color: AppColors.grey,
                        fontSize: 12.0,
                      ),
                    ),
                    const SpaceHeight(18.0),
                    const Divider(),
                    const SpaceHeight(30.0),
                    Text(
                      DateTime.now().toFormattedDate(),
                      style: const TextStyle(
                        fontWeight: FontWeight.w600,
                        color: AppColors.grey,
                      ),
                    ),
                    const SpaceHeight(6.0),
                    Text(
                      '${DateTime(2024, 3, 14, 8, 0).toFormattedTime()} - ${DateTime(2024, 3, 14, 16, 0).toFormattedTime()}',
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 20.0,
                      ),
                    ),
                  ],
                ),
              ),
              const SpaceHeight(80.0),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: GridView(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    crossAxisSpacing: 16.0,
                    mainAxisSpacing: 16.0,
                  ),
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  children: [
                    BlocBuilder<GetCompanyBloc, GetCompanyState>(
                      builder: (context, state) {
                        final latitudePoint = state.maybeWhen(
                          orElse: () => 0.0,
                          success: (data) => double.parse(data.latitude!),
                        );
                        final longitudePoint = state.maybeWhen(
                          orElse: () => 0.0,
                          success: (data) => double.parse(data.longitude!),
                        );

                        final radiusPoint = state.maybeWhen(
                          orElse: () => 0.0,
                          success: (data) => double.parse(data.radiusKm!),
                        );
                        return BlocConsumer<IsCheckedinBloc, IsCheckedinState>(
                          listener: (context, state) {
                            //
                          },
                          builder: (context, state) {
                            final isCheckin = state.maybeWhen(
                              orElse: () => false,
                              success: (data) => data.isCheckedin,
                            );

                            return MenuButton(
                              label: 'Datang',
                              iconPath: Assets.icons.menu.datang.path,
                              onPressed: () async {
                                // Deteksi lokasi palsu
                                // bool isFakeLocation =
                                //     await DetectFakeLocation().detectFakeLocation();
                                // bool isFakeLocation =
                                //     (await Antifakegps().detectFakeLocation())!;
                                // bool isFakeLocation =
                                false; //await SafeDevice.isMockLocation;
                                // Jika lokasi palsu terdeteksi
                                // if (isFakeLocation) {
                                //   // Tampilkan peringatan lokasi palsu
                                //   showDialog(
                                //     context: context,
                                //     builder: (BuildContext context) {
                                //       return AlertDialog(
                                //         title: const Text('Fake Location Detected'),
                                //         content: const Text(
                                //             'Please disable fake location to proceed.'),
                                //         actions: <Widget>[
                                //           TextButton(
                                //             child: const Text('OK'),
                                //             onPressed: () {
                                //               Navigator.of(context)
                                //                   .pop(); // Tutup dialog
                                //             },
                                //           ),
                                //         ],
                                //       );
                                //     },
                                //   );
                                // } else {
                                // masuk page checkin

                                final distanceKm =
                                    RadiusCalculate.calculateDistance(
                                        latitude ?? 0.0,
                                        longitude ?? 0.0,
                                        latitudePoint,
                                        longitudePoint);

                                print('jarak radius:  $distanceKm');

                                if (distanceKm > radiusPoint) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content:
                                          Text('Anda diluar jangkauan absen'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                  return;
                                }

                                if (isCheckin) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text('Anda sudah checkin'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                } else {
                                  context.push(const AttendanceCheckinPage());
                                }

                                // }
                              },
                            );
                          },
                        );
                      },
                    ),
                    BlocBuilder<GetCompanyBloc, GetCompanyState>(
                      builder: (context, state) {
                        final latitudePoint = state.maybeWhen(
                          orElse: () => 0.0,
                          success: (data) => double.parse(data.latitude!),
                        );
                        final longitudePoint = state.maybeWhen(
                          orElse: () => 0.0,
                          success: (data) => double.parse(data.longitude!),
                        );

                        final radiusPoint = state.maybeWhen(
                          orElse: () => 0.0,
                          success: (data) => double.parse(data.radiusKm!),
                        );
                        return BlocBuilder<IsCheckedinBloc, IsCheckedinState>(
                          builder: (context, state) {
                            final isCheckout = state.maybeWhen(
                              orElse: () => false,
                              success: (data) => data.isCheckedout,
                            );
                            final isCheckIn = state.maybeWhen(
                              orElse: () => false,
                              success: (data) => data.isCheckedin,
                            );
                            return MenuButton(
                              label: 'Pulang',
                              iconPath: Assets.icons.menu.pulang.path,
                              onPressed: () async {
                                final distanceKm =
                                    RadiusCalculate.calculateDistance(
                                        latitude ?? 0.0,
                                        longitude ?? 0.0,
                                        latitudePoint,
                                        longitudePoint);

                                print('jarak radius:  $distanceKm');

                                if (distanceKm > radiusPoint) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content:
                                          Text('Anda diluar jangkauan absen'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                  return;
                                }
                                if (!isCheckIn) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text('Anda belum checkin'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                } else if (isCheckout) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text('Anda sudah checkout'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                } else {
                                  context.push(const AttendanceCheckoutPage());
                                }
                                // Deteksi lokasi palsu
                                // bool isFakeLocation =
                                //     await DetectFakeLocation().detectFakeLocation();
                                // bool isFakeLocation =
                                false; //await SafeDevice.isMockLocation;
                                // Jika lokasi palsu terdeteksi
                                // if (isFakeLocation) {
                                //   // Tampilkan peringatan lokasi palsu
                                //   showDialog(
                                //     context: context,
                                //     builder: (BuildContext context) {
                                //       return AlertDialog(
                                //         title: const Text('Fake Location Detected'),
                                //         content: const Text(
                                //             'Please disable fake location to proceed.'),
                                //         actions: <Widget>[
                                //           TextButton(
                                //             child: const Text('OK'),
                                //             onPressed: () {
                                //               Navigator.of(context)
                                //                   .pop(); // Tutup dialog
                                //             },
                                //           ),
                                //         ],
                                //       );
                                //     },
                                //   );
                                // } else {
                                //   // masuk page checkin
                                //   context.push(const AttendanceCheckoutPage());
                                // }
                              },
                            );
                          },
                        );
                      },
                    ),
                    MenuButton(
                      label: 'Izin',
                      iconPath: Assets.icons.menu.izin.path,
                      onPressed: () {
                        context.push(const PermissionPage());
                      },
                    ),
                    MenuButton(
                      label: 'Catatan',
                      iconPath: Assets.icons.menu.catatan.path,
                      onPressed: () {},
                    ),
                  ],
                ),
              ),

              // Attendance Button
              const SpaceHeight(24.0),
              faceEmbedding != null
                  ? BlocBuilder<IsCheckedinBloc, IsCheckedinState>(
                      builder: (context, state) {
                        final isCheckout = state.maybeWhen(
                          orElse: () => false,
                          success: (data) => data.isCheckedout,
                        );
                        final isCheckIn = state.maybeWhen(
                          orElse: () => false,
                          success: (data) => data.isCheckedin,
                        );
                        return BlocBuilder<GetCompanyBloc, GetCompanyState>(
                          builder: (context, state) {
                            final latitudePoint = state.maybeWhen(
                              orElse: () => 0.0,
                              success: (data) => double.parse(data.latitude!),
                            );
                            final longitudePoint = state.maybeWhen(
                              orElse: () => 0.0,
                              success: (data) => double.parse(data.longitude!),
                            );

                            final radiusPoint = state.maybeWhen(
                              orElse: () => 0.0,
                              success: (data) => double.parse(data.radiusKm!),
                            );
                            return Button.filled(
                              onPressed: () {
                                final distanceKm =
                                    RadiusCalculate.calculateDistance(
                                        latitude ?? 0.0,
                                        longitude ?? 0.0,
                                        latitudePoint,
                                        longitudePoint);

                                print('jarak radius:  $distanceKm');

                                if (distanceKm > radiusPoint) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content:
                                          Text('Anda diluar jangkauan absen'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                  return;
                                }

                                if (!isCheckIn) {
                                  context.push(const AttendanceCheckinPage());
                                } else if (!isCheckout) {
                                  context.push(const AttendanceCheckoutPage());
                                } else {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text('Anda sudah checkout'),
                                      backgroundColor: AppColors.red,
                                    ),
                                  );
                                }

                                // context.push(const SettingPage());
                              },
                              label: 'Attendance Using Face ID',
                              icon: Assets.icons.attendance.svg(),
                              color: AppColors.primary,
                            );
                          },
                        );
                      },
                    )
                  : Button.filled(
                      onPressed: () {
                        showBottomSheet(
                          backgroundColor: AppColors.white,
                          context: context,
                          builder: (context) => Container(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const SizedBox(
                                  width: 60.0,
                                  height: 8.0,
                                  child: Divider(color: AppColors.lightSheet),
                                ),
                                const CloseButton(),
                                const Center(
                                  child: Text(
                                    'Oops !',
                                    style: TextStyle(
                                      fontWeight: FontWeight.w700,
                                      fontSize: 24.0,
                                    ),
                                  ),
                                ),
                                const SpaceHeight(4.0),
                                const Center(
                                  child: Text(
                                    'Aplikasi ingin mengakses Kamera',
                                    style: TextStyle(
                                      fontWeight: FontWeight.w700,
                                      fontSize: 15.0,
                                    ),
                                  ),
                                ),
                                const SpaceHeight(36.0),
                                Button.filled(
                                  onPressed: () => context.pop(),
                                  label: 'Tolak',
                                  color: AppColors.secondary,
                                ),
                                const SpaceHeight(16.0),
                                Button.filled(
                                  onPressed: () {
                                    context.pop();
                                    context.push(
                                        const RegisterFaceAttendencePage());
                                  },
                                  label: 'Izinkan',
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                      label: 'Attendance Using Face ID',
                      icon: Assets.icons.attendance.svg(),
                      color: AppColors.red,
                    ),
            ],
          ),
        ),
      ),
    );
  }
}
