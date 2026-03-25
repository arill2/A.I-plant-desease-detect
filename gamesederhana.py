from libs import welcome_message
import random

welcome_message('Selamat Datang di Game tebak tanam sawit!')
nama = input('Masukkan nama kamu: ')
print(f'Halo, {nama}! Ayo kita mulai bermain! pilihlah salah satu dari 4 pot tanam sawit di bawah ini untuk menanam sawit! buat pilihan mu benar agar tidak terjadi banjir')
while True:
    tanam_position = random.randint(1, 4)
    
    bentuk_pot = '[___]'
    pot_kosong = [bentuk_pot] * 4 
    pot = pot_kosong.copy() 
    pot[tanam_position - 1] = '[🌴]'

    print(f"\nCoba perhatikan pot ini: {' '.join(pot_kosong)}")

    # --- LOGIKA INPUT ---
    while True:
        pilihan_user = input('Pilih Pot yang ingin kamu tanami sawit 1 / 2 / 3 / 4 : ')
        
        # Validasi: Cek apakah input kosong atau bukan angka
        if pilihan_user == '' or not pilihan_user.isdigit():
            print("Peringatan: Masukkan angka 1 sampai 4 saja!")
            continue
        
        pilihan_user = int(pilihan_user)
        
        # Cek apakah angka di antara 1 - 4
        if 1 <= pilihan_user <= 4:
            break
        else:
            print("Pilihan hanya tersedia 1, 2, 3, atau 4!")

    # --- CEK PEMENANG ---
    if pilihan_user == tanam_position:
        print(f"Selamat! Pilihan kamu benar! Sawit tumbuh di {' '.join(pot)}.")
    else:
        print(f"Maaf kamu kalah, Sawit tumbuh di {' '.join(pot)}, bukan di nomor {pilihan_user} dan daerah kita mengalami banjir💧.")

    # --- FITUR MAIN LAGI ---
    play_again = input('\nApakah anda ingin kembali bermain [y/n]: ').lower()
    if play_again == 'n':
        break



