"""Test básico para verificar que el entorno está configurado correctamente."""


def test_import():
    """Verifica que el paquete se puede importar."""
    import aragnia

    assert aragnia is not None


def test_basic():
    """Test básico de sanidad."""
    assert True
